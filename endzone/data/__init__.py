"""Databricks SQL and Volume client.

``warehouse.py`` is the client. ``warehouse_ready.py`` starts the SQL
warehouse and waits until it is RUNNING. ``retry_command.py`` reruns a
command after a transient Databricks error. Bronze ingest lives in
``endzone.ingest``.
"""
