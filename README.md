## Oracle RAG Assistant

This repository contains a prototype **Retrieval‑Augmented Generation (RAG) assistant** for Oracle databases.  The tool implements a do‑it‑yourself (DIY) RAG pipeline using open‑source building blocks instead of proprietary hosted services.  It is designed to provide column‑level profiling, dataset comparison and natural‑language question answering over an Oracle schema.

### Features

* **Oracle connector** – uses [`python‑oracledb`](https://pypi.org/project/oracledb/) to establish a JDBC‑like connection to an Oracle instance.  The `OracleConnector` class wraps common tasks such as connecting, reading table schemas and sampling rows.
* **Data profiling** – the `DataProfiler` class builds summary statistics (count, nulls, distinct values, min/max, etc.) for each column in a table.  Summaries are returned as `polars.DataFrame` objects for efficient processing.
* **Column comparison** – `ColumnComparator` performs deep comparison of two tables or views.  It checks schema differences (names, types) and also reports distribution differences when the column types match.
* **DIY RAG pipeline** – `RagAssistant` generates descriptive strings for every table and column, embeds those descriptions using a [Sentence‑Transformers](https://www.sbert.net/) model and stores the vectors in a [FAISS](https://github.com/facebookresearch/faiss) index.  When a question is posed, it embeds the query, performs similarity search, constructs a context prompt and returns it.  You can plug any large‑language model (LLM) in the placeholder stub to turn the context into SQL queries or natural‑language responses.

### Repository contents

| Path | Description |
|---|---|
| `oracle_rag_tool.py` | Source code implementing the `OracleConnector`, `DataProfiler`, `ColumnComparator` and `RagAssistant` classes. |

### Installation

1. Ensure you have Python ≥3.9 installed.
2. Install the dependencies.  You can use the following command in the root of this repository:

   ```bash
   pip install -r requirements.txt
   ```

   If you plan to use GPU acceleration with FAISS, replace `faiss-cpu` in the requirements with `faiss-gpu`.

3. Set up an Oracle database connection.  The `OracleConnector` expects a connection string in the form `user/password@host:port/service_name`.  You can also pass a dictionary of `cx_Oracle` parameters.

### Usage

Below is a basic example of how to profile a table and ask a question:

```python
from oracle_rag_tool import OracleConnector, DataProfiler, RagAssistant

# Connect to your Oracle instance
conn_str = "scott/tiger@localhost:1521/XEPDB1"
conn = OracleConnector(conn_str)

# Profile a table
profiler = DataProfiler(conn)
summary_df = profiler.profile_table('EMP')
print(summary_df)

# Build the RAG index over the schema
rag = RagAssistant(conn)
rag.ingest_schema()

# Ask a question about your data
response = rag.answer_question("How many employees were hired in 2023?")
print(response)
```

### License

This project is provided for educational purposes.  Feel free to modify and extend it to suit your needs.