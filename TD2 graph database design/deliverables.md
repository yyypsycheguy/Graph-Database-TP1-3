# Deliverables
## Screenshot of Neo4j Browser showing:
Constraints
At least 3 queries with output
![Image 1](photos/1EC121F5-AB71-4429-91FD-20F28A523FF9.jpeg)
![Image 3](photos/A36A4648-2FE2-4D3A-9ED4-EAD7C22844E5.jpeg)
![Image 4](photos/A6407DC1-2461-4E81-B410-5758EE184EC6.jpeg)
![Image 5](photos/ED4B3A9D-24BE-4B70-8DD9-20418772EBC3.jpeg)

## Output of:
```bash
curl "http://localhost:8000/health"
```
![Image 2](photos/37D4695F-6567-4610-BCAC-233F5C7C3AE3.jpeg)

## Short note:
1) Quick Take on Recs and Going Production
What kind of recommendation trick can you pull off here?
Since the project is about Customers, Products, and those interaction events like views and clicks, the main thing we can do is Collaborative Filtering. Wwe can easily figure out which products people are checking out together (Item-Item CF) by using Neo4j's graph algorithms like co-occurrence and similarity, like Jaccard or cosine similarity. Then we could run PageRank to see which products are the most popular overall and recommend those as a starting point.

2) If you wanted to make this mini-project ready for a real company, what would you change?
To take this from a cool project to a real system, the first thing is stop reloading all the data every time. I should set up incremental ETL, only grabbing the new stuff from Postgres so the data loads way faster. We should also separate the data loading process from the app itself and put in a cache before the FastAPI app to serve those recommendations super quick without always hitting Neo4j.