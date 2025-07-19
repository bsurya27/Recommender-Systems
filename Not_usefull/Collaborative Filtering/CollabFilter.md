# Collaborative Filtering

Collaborative Filtering means multiple people collaborating to suggest/filter out useful documents.

Indexer: Reads documents from external sources such as electronic mail, NetNews, or newswires and adds them to the document store. The indexer is responsible for parsing documents into a set of indexed fields that can be referenced in queries.

Document store: Provides long-term storage for all documents. It also maintains indexes on the stored documents so that queries over the document database can be efficiently executed. The document store is append-only.

Annotation Store: Provides long-term storage for all annotations associated with the documents. It is append-only

Filterer: Repeatedly runs the batches of user given queries. Matching documents are placed in a little box owned by a user

Little Box: Queues up documents of interest to a particular user. Each user has a little box, where documents are deposited by the filterer and removed by a user's document reader.

Remailer: Periodically sends the contents of a user's little box to the user via electronic mail.

Appraiser: Applies personalized classification to a user's documents (i.e., to those documents in the user's little box). This function can automatically prioritize and categorize documents.

Reader~Browser. Provides the user interface for accessing services. This includes facilities for such tasks as adding/deleting/editing filters, retrieving new documents, displaying documents, organizing documents into folders, supplying annotations, and running ad hoc queries
