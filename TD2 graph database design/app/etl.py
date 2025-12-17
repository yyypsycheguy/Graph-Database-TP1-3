from pathlib import Path
import time
import psycopg2
from neo4j import GraphDatabase
from fastapi import HTTPException


def wait_for_postgres():
    """Wait until PostgreSQL is ready to accept connections."""
    for attempt in range(10):
        try:
            conn = psycopg2.connect(
                host="postgres_db",
                port=5432,
                dbname="your_database",
                user="your_user",
                password="your_password"
            )
            conn.close()
            print("✅ PostgreSQL is ready.")
            return
        except Exception as e:
            print(f"⏳ Waiting for PostgreSQL... (attempt {attempt + 1}/10):", e)
            time.sleep(3)
    raise HTTPException(status_code=500, detail="PostgreSQL not ready after multiple attempts.")


def wait_for_neo4j():
    """Wait until Neo4j is ready to accept connections."""
    for attempt in range(10):
        try:
            driver = GraphDatabase.driver(
                "bolt://neo4j_db:7687",
                auth=("neo4j", "your_password")
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            print("✅ Neo4j is ready.")
            return
        except Exception as e:
            print(f"⏳ Waiting for Neo4j... (attempt {attempt + 1}/10):", e)
            time.sleep(3)
    raise HTTPException(status_code=500, detail="Neo4j not ready after multiple attempts.")


def etl():
    """
    Main ETL function that migrates data from PostgreSQL to Neo4j.
    
    This function performs the complete Extract, Transform, Load process:
    1. Waits for both databases to be ready
    2. Sets up Neo4j schema using queries.cypher file
    3. Extracts data from PostgreSQL tables
    4. Transforms relational data into graph format
    5. Loads data into Neo4j with appropriate relationships
    """
    # Ensure dependencies are ready (useful when running in docker-compose)
    wait_for_postgres()
    wait_for_neo4j()

    # Get path to your Cypher schema file
    queries_path = Path(__file__).with_name("queries.cypher")

    # Set up Neo4j schema
    neo4j_driver = GraphDatabase.driver(
        "bolt://neo4j_db:7687",
        auth=("neo4j", "your_password")
    )

    with open(queries_path, 'r') as file:
        queries = file.read()
        for query in queries.split(';'):
            if query.strip():
                with neo4j_driver.session() as session:
                    session.run(query)
    print("✅ Neo4j schema initialized.")

    # Extracts data from PostgreSQL tables
    pg_conn = psycopg2.connect(
        host="postgres_db",
        port=5432,
        dbname="your_database",
        user="your_user",
        password="your_password"
    )
    pg_cursor = pg_conn.cursor()

    tables = ["categories", "products", "customers", "orders", "order_items"]
    data = {}
    for table in tables:
        pg_cursor.execute(f"SELECT * FROM {table};")
        columns = [desc[0] for desc in pg_cursor.description]
        rows = pg_cursor.fetchall()
        data[table] = [dict(zip(columns, row)) for row in rows]
    print("✅ Data extracted from PostgreSQL.")

    # Transforms relational data into graph format
    categories = data["categories"]
    products = data["products"]
    customers = data["customers"]
    orders = data["orders"]
    order_items = data["order_items"]

    # Loads data into Neo4j
    with neo4j_driver.session() as session:
        # Create Category nodes
        for c in categories:
            session.run(
                "MERGE (cat:Category {id: $id, name: $name})",
                {"id": c["id"], "name": c["name"]}
            )

        # Create Product nodes and IN_CATEGORY relationships
        for p in products:
            session.run(
                """
                MERGE (p:Product {id: $id, name: $name, price: $price})
                WITH p
                MATCH (c:Category {id: $category_id})
                MERGE (p)-[:IN_CATEGORY]->(c)
                """,
                {
                    "id": p["id"],
                    "name": p["name"],
                    "price": p["price"],
                    "category_id": p["category_id"],
                },
            )

        # Create Customer nodes
        for cust in customers:
            session.run(
                "MERGE (c:Customer {id: $id, name: $name, join_date: $join_date})",
                {
                    "id": cust["id"],
                    "name": cust["name"],
                    "join_date": cust["join_date"],
                },
            )

        # Create Order nodes and PLACED relationships
        for order in orders:
            session.run(
                """
                MERGE (o:Order {id: $id, order_date: $order_date})
                WITH o
                MATCH (c:Customer {id: $customer_id})
                MERGE (c)-[:PLACED]->(o)
                """,
                {
                    "id": order["id"],
                    "order_date": order["order_date"],
                    "customer_id": order["customer_id"],
                },
            )

        # Create CONTAINS relationships between Orders and Products
        for item in order_items:
            session.run(
                """
                MATCH (o:Order {id: $order_id}), (p:Product {id: $product_id})
                MERGE (o)-[r:CONTAINS {quantity: $quantity}]->(p)
                """,
                {
                    "order_id": item["order_id"],
                    "product_id": item["product_id"],
                    "quantity": item["quantity"],
                },
            )

    pg_cursor.close()
    pg_conn.close()
    neo4j_driver.close()

    print("🎉 ETL process completed successfully.")
    return {"status": "success", "message": "ETL completed successfully"}
