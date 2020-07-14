## Questions/Topics for clarification before starting the DE phase

### Business Context
1. What is the problem we are trying to solve and how the client will benefit from it?
   - understanding the problem is crucial for the next stages to see the data from the problem solving perspective
2. Make sure that the problem from ML perspective is well defined
   - what kind of model we are going to create - supervised/unsupervised, regression/classification/other
   - what is the impact that the prediction can have - is it better to have false positives or false negatives?
3. Try to understand the workflow of data
   - what are the possible sources of data? Is it just gathered from one place? Or there are multiple services that collect **different** data?
***
### Data access
1. Check any legal obstacles - do we need/have NDA signed?
2. Do we have everything needed to connect to the data?
   - check if the **data** can be accessed i.e. there are no access restriction on tables for given user etc
3. Can we work on the provided infrastructure or it is just read-only?
   - if we can work on it - make sure it won't be updated or overwritten in order to avoid loosing progress
   - if we need to create our environment - make sure the setup has the same version of the software i.e. PostgreSQL 10 vs 11
***
### Prediction Definition
1. Do we have clear definition of what are the possible classes for the prediction?
2. If yes - do we know how to distinguish classes in the available dataset? 
   - what is the column that says what is the class
3. If no - we need to either gather or calculate it ourselves
   - If gather - where from? Do we need data annotation process or something?
   - If calculate - do we know how? What are the thresholds for given classes, what should be taken into account etc
***
### Data Sanity Check
1. Do we have access to the full dataset?
   - if not, suggest that it would be for mutual benefit if we could have the whole dataset - we can always subset it ourselves if it is too big for initial analysis
   - it is not crucial to have the whole dataset - client might want to stick with a portion of data for model validation i.e.
   - if we are working on sample data - make sure to check below points about sample dataset to avoid any misunderstandings
2. Any other sources of data that could come in handy?
   - if yes - can we get access or get some samples to have the whole picture
   - if there are no some serious obstacles - it is the best to have access to everything upfront rather than trying to add new data in the middle of the process
3. What is the full data volume? 
   - GB/TB/rows
4. Do we have samples for each output class? 
   - no 'ghost' classes without samples
5. If we got only sample data - how is the full dataset going to be shared?
   - the same way we got sample or it is going to be different?
   - if some other way - we need to know how to prepare for the later data acquisition to smoother out the process
6. If we got only sample data - how was the full dataset sampled? 
   - does it represent the full dataset accordingly?
   - is the class distribution similar to full dataset?
   - are we dealing with balanced or unbalanced data?
7. Any data enrichment done before? 
   - any sources they tried or acquired to enrich the data?
8. Are we working with production-like data structure?
   - if we are working with some legacy data and the current production one has different structure that might create a problem when trying to implement the model on production with different data structure
***
### Data Quality Check
1. How much deduplication is needed?
   - are there lookup tables with duplicated entries?
   - can we do the deduplication by ourselves or it requires specific know-how on how to map things?
   - do we have access to some golden source of data that we can use for mapping?
2. Is the data normalized?
   - if not it might take a bit longer to either normalize everything or make sure that the values are kept in sync or come from some specific distribution
3. Are uuids, primary and foreign keys set on all tables?
   - if not we might need some guideline on how things are interconnected
   - we can always try our "best guess" but that needs to be confirmed with the client to check we are doing the right thing
4. How much legacy data is there?
   - i.e columns that are no longer used/filled/updated because of architectural or business changes
   - can they be easily spotted? Are there any distinctions in the naming convention?
   - if not do we have some clear way of figuring out which ones are up-to-date
5. Columns format consistency
   - are values kept in the same, standardized format or there are mixed convention used?
6. Contradicting data
   - that is probably linked to the legacy data problem
   - i.e. cases where there seem to be many ways of getting the same information but depending on the mapping we can get different results
