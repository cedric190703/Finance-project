"""A compact desk dashboard backed by the FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import httpx
import plotly.express as px
import polars as pl
import streamlit as st

API_URL = os.environ.get("AEGIS_API_URL", "http://localhost:8000")


def _get(path: str, params: dict[str, str] | None = None) -> Any:
    """Fetch a JSON API response and surface a readable Streamlit error."""
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=10.0)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Render the operational, risk and P&L views."""
    st.set_page_config(page_title="Aegis", page_icon="🛡️", layout="wide")
    st.title("Aegis — Risk & P&L")
    st.caption(f"API: {API_URL}")
    try:
        health = _get("/health")
        st.success(f"Service {health['status']} · version {health['version']}")
        coverage = pl.DataFrame(_get("/store/coverage"))
    except httpx.HTTPError as error:
        st.error(f"The API is unavailable: {error}")
        st.stop()

    left, right = st.columns((1, 2))
    with left:
        st.subheader("Warehouse coverage")
        st.dataframe(coverage, use_container_width=True, hide_index=True)
    with right:
        st.subheader("P&L waterfall")
        st.info("Select two dates after the corresponding market snapshots are available.")
        start_date = st.date_input("Opening date")
        end_date = st.date_input("Closing date")
        if st.button("Explain P&L"):
            try:
                payload = _get(
                    "/pnl/explain",
                    {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
                )
                components = pl.DataFrame(payload["components"])
                st.metric("Actual P&L", f"{payload['total_pnl']:,.0f} USD")
                st.plotly_chart(
                    px.bar(
                        components,
                        x="component",
                        y="pnl",
                        color="pnl",
                        color_continuous_scale="RdYlGn",
                    ),
                    use_container_width=True,
                )
            except httpx.HTTPError as error:
                st.error(f"Unable to explain P&L: {error}")


if __name__ == "__main__":
    main()
