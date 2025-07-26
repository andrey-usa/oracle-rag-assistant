"""
oracle_rag_tool.py
====================

This module provides a set of classes and functions to build the foundation
for a Retrieval‑Augmented Generation (RAG) assistant that works with Oracle
databases.  It demonstrates how to connect to Oracle using the modern
``python‑oracledb`` driver, fetch data into Polars DataFrames for
high‑performance analytics, perform basic data profiling, compare columns
across tables, and lays the groundwork for integrating a large language
model (LLM) for natural‑language SQL generation and data quality validation.

The code herein is intended as a starting point; it assumes that callers
have installed the required dependencies (``polars``, ``python‑oracledb``,
``pyarrow``, and optionally ``sentence‑transformers`` or ``faiss`` if
embedding‑based search is desired).  It also shows how to build a simple
in‑memory vector store for retrieval.

Example usage::

    from oracle_rag_tool import OracleConnector, DataProfiler, ColumnComparator

    # Connect to an Oracle database using python-oracledb.  Credentials
    # should be provided via environment variables or a secure secrets manager.
    connector = OracleConnector("user", "password", "hostname:1521/dbname")
    df = connector.query_to_polars("SELECT * FROM employees")

    # Perform profiling on the DataFrame.
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(df)
    print(profile)

    # Compare columns between two DataFrames.
    df2 = connector.query_to_polars("SELECT * FROM employees_backup")
    comparator = ColumnComparator()
    comparison = comparator.compare_dataframes(df, df2)
    print(comparison)

    # Build a RAG assistant (requires embeddings and LLM integration).  See
    # class docstring for more details.

This file is placed in the shared ``/home/oai/share`` directory for easy
access.  It does not execute any code on import so it is safe to use as a
library.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import polars as pl
except ImportError as e:
    raise ImportError(
        "Polars is required for oracle_rag_tool. Install with `pip install polars`"
    ) from e

# python‑oracledb is used to connect to Oracle databases.  It is a thin
# wrapper around Oracle's network protocols and supports efficient fetching
# of results directly into Arrow/Polars.  See the official documentation
# for details: https://oracle.github.io/python-oracledb/.
try:
    import oracledb
except ImportError:
    oracledb = None  # type: ignore

try:
    import pyarrow as pa  # type: ignore
except ImportError:
    pa = None  # type: ignore

try:
    # sentence-transformers provides easy access to pretrained embedding
    # models.  It is optional and can be installed with `pip install
    # sentence-transformers`.
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    # faiss is used here as an example vector store.  It is optional; you
    # could replace it with Qdrant, Pinecone, Milvus, etc.  Install
    # `faiss-cpu` or `faiss-gpu` as appropriate.
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

logger = logging.getLogger(__name__)


class OracleConnector:
    """Connects to an Oracle database and fetches results into Polars.

    The connector uses the ``python-oracledb`` driver if available.  It
    encapsulates the creation of connections and cursors and exposes a
    convenience method to fetch query results into a Polars ``DataFrame``.

    Attributes
    ----------
    user : str
        Oracle username.
    password : str
        Oracle password.
    dsn : str
        Data Source Name or connect string.  This can be in the easy
        connect format ``host:port/service`` or a TNS alias defined in
        ``tnsnames.ora``.
    config : Optional[Dict[str, Any]]
        Additional parameters passed to ``oracledb.connect()`` such as
        ``mode`` or ``encoding``.
    """

    def __init__(self, user: str, password: str, dsn: str, **config: Any) -> None:
        if oracledb is None:
            raise RuntimeError(
                "python-oracledb is not installed. Please install it with `pip install python-oracledb`"
            )
        self.user = user
        self.password = password
        self.dsn = dsn
        self.config = config
        logger.debug("OracleConnector initialized with DSN %s", dsn)

    def _get_connection(self) -> "oracledb.Connection":
        """Create and return a new Oracle connection.

        Using a new connection for each query helps avoid issues with
        concurrency and ensures that fetch options are applied per query.
        """
        conn = oracledb.connect(user=self.user, password=self.password, dsn=self.dsn, **self.config)
        return conn

    def query_to_polars(self, sql: str, parameters: Optional[Dict[str, Any]] = None, arraysize: int = 1000) -> pl.DataFrame:
        """Execute a SQL statement and return the results as a Polars DataFrame.

        Parameters
        ----------
        sql : str
            The SQL query to execute.
        parameters : Optional[Dict[str, Any]]
            Optional bind variables for parameterized queries.
        arraysize : int
            The number of rows to fetch per network round‑trip.  Adjust
            according to your network latency and memory constraints.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame containing the query results.
        """
        logger.debug("Running SQL: %s", sql)
        with self._get_connection() as conn:
            # python-oracledb offers fetch_df_all() which returns an internal
            # DataFrame backed by Apache Arrow.  The documentation notes
            # that this improves performance and memory usage when working
            # with DataFrame libraries【292529163184194†L83-L89】.
            df = conn.fetch_df_all(statement=sql, parameters=parameters, arraysize=arraysize)
            # Convert the internal DataFrame to a Polars DataFrame.  This uses
            # pyarrow under the hood via polars.from_arrow().
            if pa is None:
                raise RuntimeError(
                    "pyarrow is required to convert Oracle DataFrames to Polars. Install with `pip install pyarrow`"
                )
            arrow_table = pa.Table.from_pydict({name: df.get_column_by_name(name) for name in df.column_names()})
            pl_df = pl.from_arrow(arrow_table)
            logger.debug("Fetched %d rows and %d columns", pl_df.height, pl_df.width)
            return pl_df


class DataProfiler:
    """Computes basic statistics and data quality indicators for Polars DataFrames.

    This class provides methods to compute summary statistics such as
    minimum/maximum values, mean and median for numeric columns, counts of
    missing values, and cardinality of categorical columns.  It is designed
    to be extensible—additional profiling metrics can easily be added.
    """

    def profile_dataframe(self, df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Return a profiling report for each column in a DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            The Polars DataFrame to profile.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A mapping from column name to a dictionary of computed metrics.
        """
        profile: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            series = df[col]
            col_stats: Dict[str, Any] = {}
            col_stats["dtype"] = str(series.dtype)
            # Missing values
            col_stats["missing_count"] = series.is_null().sum()
            col_stats["missing_percentage"] = (
                float(col_stats["missing_count"]) / df.height * 100 if df.height > 0 else 0.0
            )
            # Cardinality
            col_stats["n_unique"] = series.n_unique()
            # Numeric statistics
            if pl.datatypes.is_numeric_dtype(series.dtype):
                col_stats["min"] = series.min()
                col_stats["max"] = series.max()
                col_stats["mean"] = float(series.mean()) if series.len() > 0 else None
                col_stats["median"] = float(series.median()) if series.len() > 0 else None
                col_stats["std"] = float(series.std()) if series.len() > 1 else None
            # String or categorical statistics
            elif pl.datatypes.is_string_dtype(series.dtype) or pl.datatypes.is_categorical_dtype(series.dtype):
                # Determine the most common values and their counts
                value_counts = series.value_counts().sort("counts", descending=True)
                top_values = (
                    value_counts.head(5)
                    .to_dict(as_series=False)
                    if not value_counts.is_empty() else {"": []}
                )
                col_stats["top_values"] = top_values
            profile[col] = col_stats
        return profile


class ColumnComparator:
    """Compares two Polars DataFrames at the column level.

    The comparator inspects the schema of each DataFrame, compares data types,
    and optionally computes overlap statistics such as value set differences
    for categorical columns or distribution differences for numeric columns.  It
    returns a structured report highlighting columns that match, columns that
    differ, and suggestions for reconciliation.
    """

    def compare_dataframes(
        self, df1: pl.DataFrame, df2: pl.DataFrame, check_values: bool = True
    ) -> Dict[str, Any]:
        """Compare two DataFrames and return a report.

        Parameters
        ----------
        df1, df2 : pl.DataFrame
            The DataFrames to compare.
        check_values : bool, optional
            Whether to compare the actual values of matching columns.  If
            ``True``, the function will compute differences in value sets
            (for categorical columns) and distribution statistics (for
            numeric columns).  This can be expensive for large data sets.

        Returns
        -------
        Dict[str, Any]
            A report containing matched columns, mismatched columns,
            and optionally value differences.
        """
        report: Dict[str, Any] = {
            "matched_columns": [],
            "type_mismatches": [],
            "left_only": [],
            "right_only": [],
            "value_differences": {},
        }
        # Compare schemas
        cols1 = {col: dtype for col, dtype in zip(df1.columns, df1.dtypes)}
        cols2 = {col: dtype for col, dtype in zip(df2.columns, df2.dtypes)}
        for col in cols1.keys() | cols2.keys():
            in1 = col in cols1
            in2 = col in cols2
            if in1 and in2:
                if cols1[col] == cols2[col]:
                    report["matched_columns"].append(col)
                    # Optionally compare values
                    if check_values:
                        s1 = df1[col]
                        s2 = df2[col]
                        if pl.datatypes.is_numeric_dtype(s1.dtype):
                            # Compute difference in mean and std deviation
                            diff = {
                                "mean_diff": float(s1.mean() - s2.mean()),
                                "std_diff": float(s1.std() - s2.std()),
                            }
                            report["value_differences"][col] = diff
                        elif pl.datatypes.is_string_dtype(s1.dtype) or pl.datatypes.is_categorical_dtype(s1.dtype):
                            # Compare value sets
                            set1 = set(s1.unique())
                            set2 = set(s2.unique())
                            report["value_differences"][col] = {
                                "left_only_values": list(set1 - set2),
                                "right_only_values": list(set2 - set1),
                            }
                else:
                    report["type_mismatches"].append(
                        {
                            "column": col,
                            "left_dtype": str(cols1[col]),
                            "right_dtype": str(cols2[col]),
                        }
                    )
            elif in1:
                report["left_only"].append(col)
            else:
                report["right_only"].append(col)
        return report


@dataclasses.dataclass
class VectorStore:
    """A simple in‑memory vector store using FAISS for similarity search.

    This helper class wraps FAISS to build an index of embeddings.  It can
    be replaced with other vector database implementations such as Qdrant,
    Milvus, or Pinecone.  The vector store stores tuples of (id, metadata)
    alongside the embedding vectors.
    """

    dimension: int
    ids: List[str] = dataclasses.field(default_factory=list)
    metadata: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    index: Any = None  # FAISS index

    def __post_init__(self) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed. Install `faiss-cpu` or `faiss-gpu` to use VectorStore."
            )
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.debug("Initialized FAISS index with dimension %d", self.dimension)

    def add(self, embeddings: List[List[float]], ids: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings to the store.

        Parameters
        ----------
        embeddings : List[List[float]]
            List of embedding vectors.
        ids : List[str]
            Identifiers corresponding to each embedding.
        metadata : List[Dict[str, Any]]
            Arbitrary metadata associated with each embedding.
        """
        if len(embeddings) != len(ids) or len(ids) != len(metadata):
            raise ValueError("Embeddings, ids and metadata must have the same length")
        self.index.add(np.array(embeddings, dtype="float32"))  # type: ignore[name-defined]
        self.ids.extend(ids)
        self.metadata.extend(metadata)
        logger.debug("Added %d embeddings to the vector store", len(ids))

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for the nearest neighbors to a query embedding.

        Parameters
        ----------
        query_embedding : List[float]
            The embedding vector representing the query.
        top_k : int, optional
            Number of nearest neighbors to return.

        Returns
        -------
        List[Tuple[str, Dict[str, Any], float]]
            A list of tuples containing the id, metadata and similarity score for
            each of the top_k nearest embeddings.
        """
        query_vec = np.array([query_embedding], dtype="float32")  # type: ignore[name-defined]
        scores, indices = self.index.search(query_vec, top_k)
        results: List[Tuple[str, Dict[str, Any], float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.ids):
                results.append((self.ids[idx], self.metadata[idx], float(score)))
        return results


class RagAssistant:
    """A skeleton RAG assistant for Oracle.

    This class demonstrates how a retrieval‑augmented generation system could be
    wired together for querying and reasoning about data stored in an Oracle
    database.  It is intentionally generic: you can plug in different
    embedding models, vector stores, and language models to meet your needs.

    Workflow:

    1. **Ingest:** Fetch schema information, column descriptions, sample
       values, and other relevant data from the Oracle database.  Turn
       these into textual documents that describe each table/column.
    2. **Embed:** Use an embedding model (e.g. from sentence‑transformers)
       to convert these documents into dense vectors and store them in a
       vector store for fast similarity search.
    3. **Retrieve:** Given a user question, embed the query and search the
       vector store to find the most relevant tables, columns, or samples.
    4. **Generate:** Feed the retrieved context and the user question into
       an LLM (e.g. GPT‑4 or other locally hosted model) to generate SQL
       statements or answer data quality questions.

    Best practices for RAG, such as using semantic search instead of keyword
    matching and predefining SQL templates for common queries, help reduce
    latency and improve accuracy【472195643798474†L18-L96】.  Retrieval should
    leverage embeddings to allow semantic matching between user prompts and
    database content【472195643798474†L75-L83】.  Query optimization via
    templates can further enhance performance【472195643798474†L92-L100】.

    Note: This implementation leaves the actual LLM call unimplemented.
    You can integrate OpenAI's API, local models via Hugging Face, or
    open‑source models through LangChain or LlamaIndex.  Always ensure
    compliance with your organization's security policies when sending
    database schema information to external services.
    """

    def __init__(
        self,
        connector: OracleConnector,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dim: int = 384,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for RagAssistant. Install with `pip install sentence-transformers`"
            )
        if faiss is None:
            raise RuntimeError(
                "faiss is required for RagAssistant. Install with `pip install faiss-cpu`"
            )
        self.connector = connector
        # Load embedding model
        self.model = SentenceTransformer(embedding_model_name)
        self.vector_store = VectorStore(dimension=vector_dim)
        logger.debug("RagAssistant initialized with embedding model %s", embedding_model_name)

    def ingest_schema(self, table_names: Optional[List[str]] = None, sample_rows: int = 5) -> None:
        """Ingest schema and sample data from Oracle into the vector store.

        Parameters
        ----------
        table_names : Optional[List[str]]
            List of table names to ingest.  If None, the assistant queries
            ``user_tab_columns`` and ``user_tables`` to discover all tables.
        sample_rows : int
            Number of sample rows to fetch for each table for context.
        """
        # Discover tables if not provided
        if table_names is None:
            sql_tables = "SELECT table_name FROM user_tables"
            tables_df = self.connector.query_to_polars(sql_tables)
            table_names = tables_df["TABLE_NAME"].to_list() if "TABLE_NAME" in tables_df.columns else []
        for table in table_names:
            # Get column metadata
            sql_cols = (
                "SELECT column_name, data_type, data_length FROM user_tab_columns "
                "WHERE table_name = :table"
            )
            cols_df = self.connector.query_to_polars(sql_cols, parameters={"table": table})
            # Fetch sample rows
            sql_sample = f"SELECT * FROM {table} FETCH FIRST {sample_rows} ROWS ONLY"
            try:
                sample_df = self.connector.query_to_polars(sql_sample)
            except Exception as e:
                logger.warning("Failed to fetch sample rows for %s: %s", table, e)
                sample_df = pl.DataFrame()
            # Build document text
            doc = self._build_table_document(table, cols_df, sample_df)
            embedding = self.model.encode(doc).tolist()
            self.vector_store.add([embedding], [table], [
                {"table": table, "columns": cols_df.to_dict(as_series=False), "sample": sample_df.head(sample_rows).to_dict(as_series=False)}
            ])
            logger.debug("Ingested table %s with %d columns", table, cols_df.height)

    def _build_table_document(self, table: str, cols_df: pl.DataFrame, sample_df: pl.DataFrame) -> str:
        """Construct a textual description of a table for embedding.

        Includes the table name, column names and types, and optionally
        a few sample rows serialized to text.  You can customize this
        representation to include comments, foreign key relationships or
        other metadata.
        """
        lines = [f"Table: {table}"]
        for row in cols_df.iter_rows():
            col_name, data_type, data_length = row
            lines.append(f"Column: {col_name} ({data_type}{f'({data_length})' if data_length else ''})")
        if sample_df.height > 0:
            lines.append("Sample rows:")
            # Limit the sample to a few rows to keep embeddings concise
            for i in range(min(sample_df.height, 3)):
                row_repr = ", ".join(f"{col}={sample_df[col][i]}" for col in sample_df.columns)
                lines.append(row_repr)
        return "\n".join(lines)

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """Answer a user question by retrieving relevant context and calling an LLM.

        Parameters
        ----------
        question : str
            The natural language question to answer (e.g. "Find all employees
            hired after 2020" or "Compare salaries by department").
        top_k : int
            Number of relevant tables to retrieve from the vector store.

        Returns
        -------
        str
            A generated answer.  At present this is a stub; replace the
            placeholder with a call to your LLM of choice.
        """
        # Embed the query
        query_embedding = self.model.encode(question).tolist()
        # Retrieve relevant tables
        retrieved = self.vector_store.search(query_embedding, top_k=top_k)
        # Build a context string from retrieved metadata
        context_parts = []
        for _id, meta, score in retrieved:
            context_parts.append(f"Table {meta['table']} (score={score:.3f}):")
            cols_desc = ", ".join(
                f"{name} ({dtype})" for name, dtype in zip(meta["columns"]["column_name"], meta["columns"]["data_type"])
            )
            context_parts.append(f"Columns: {cols_desc}")
            if meta.get("sample"):
                # Show one sample row
                sample = meta["sample"]
                if sample:
                    row_repr = ", ".join(f"{col}={sample[col][0]}" for col in sample.keys())
                    context_parts.append(f"Sample: {row_repr}")
        context = "\n".join(context_parts)
        # Here you would call the generative model, e.g. using OpenAI's API
        # or a local model.  The prompt should include the question and the
        # retrieved context.  Since external API calls are not allowed in
        # this environment, we return a placeholder response.
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        logger.debug("Generated prompt for LLM:\n%s", prompt)
        # Placeholder: return the prompt as the response for demonstration
        return prompt


# Optional dependencies that require numpy.  Import numpy lazily to avoid
# imposing a dependency on users who don't need vector search.
try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore