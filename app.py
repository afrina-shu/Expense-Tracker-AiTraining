"""💸 AI Expense Tracker - Google ADK + MCP + SQLite + Streamlit (single file).

Day 2 concepts:
  * Agent-as-Tool + LLM-driven routing : coordinator -> expense_recorder / expense_analyst
  * MCP (stdio) over SQLite            : this same file doubles as the MCP server (`python app.py --mcp`)
  * tool_filter / least privilege      : recorder can WRITE, analyst can only READ
"""
import sys

# ======================================================================================
# 1) MCP SERVER MODE  -  `python app.py --mcp`  (must run BEFORE Streamlit is imported)
# ======================================================================================
CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Health", "Entertainment", "Other"]


def run_mcp_server() -> None:
    import os
    import re
    import sqlite3
    from datetime import date
    from pathlib import Path

    try:  # mcp>=2 renamed FastMCP -> MCPServer
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server

    db = Path(os.environ.get("EXPENSE_DB", "expenses.db")).resolve()
    mcp = Server("expense-sqlite")

    def conn_rw():
        c = sqlite3.connect(db)
        c.row_factory = sqlite3.Row
        c.execute(
            "CREATE TABLE IF NOT EXISTS expenses (id INTEGER PRIMARY KEY AUTOINCREMENT, amount REAL NOT NULL, "
            "category TEXT NOT NULL, note TEXT, spent_on TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')))"
        )
        return c

    def conn_ro():  # read tools use a read-only connection
        conn_rw().close()  # make sure the table exists
        c = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA query_only = ON")
        return c

    def where(month: str, category: str):
        clauses, params = [], []
        if month:
            if not re.fullmatch(r"\d{4}-\d{2}", month):
                raise ValueError("month must look like 2026-11")
            clauses.append("substr(spent_on,1,7) = ?")
            params.append(month)
        if category:
            clauses.append("lower(category) = lower(?)")
            params.append(category)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    # ---------------- WRITE tools (only the recorder agent may call these) ----------------
    @mcp.tool()
    def add_expense(amount: float, category: str, note: str = "", spent_on: str = "") -> dict:
        """Record one expense. category must be one of: Food, Transport, Shopping, Bills, Health,
        Entertainment, Other. spent_on is YYYY-MM-DD (empty = today)."""
        if amount <= 0:
            return {"error": "amount must be greater than 0"}
        cat = next((c for c in CATEGORIES if c.lower() == category.strip().lower()), "Other")
        try:
            day = date.fromisoformat(spent_on) if spent_on else date.today()
        except ValueError:
            return {"error": "spent_on must be YYYY-MM-DD"}
        with conn_rw() as c:
            cur = c.execute(
                "INSERT INTO expenses(amount, category, note, spent_on) VALUES (?,?,?,?)",
                (round(amount, 2), cat, note[:200], day.isoformat()),
            )
            return {"saved": True, "id": cur.lastrowid, "amount": round(amount, 2), "category": cat,
                    "note": note, "spent_on": day.isoformat()}

    @mcp.tool()
    def delete_expense(expense_id: int) -> dict:
        """Delete one expense by its id."""
        with conn_rw() as c:
            n = c.execute("DELETE FROM expenses WHERE id = ?", (expense_id,)).rowcount
        return {"deleted": bool(n), "id": expense_id}

    # ---------------- READ tools (analyst agent) ----------------
    @mcp.tool()
    def list_expenses(limit: int = 20, category: str = "", month: str = "") -> list[dict]:
        """Newest expenses first. Optional filters: category, month as YYYY-MM."""
        w, p = where(month, category)
        with conn_ro() as c:
            rows = c.execute(f"SELECT * FROM expenses{w} ORDER BY spent_on DESC, id DESC LIMIT ?", (*p, min(limit, 100)))
            return [dict(r) for r in rows]

    @mcp.tool()
    def get_total(month: str = "", category: str = "") -> dict:
        """Total amount and number of expenses. Optional filters: month YYYY-MM, category."""
        w, p = where(month, category)
        with conn_ro() as c:
            r = c.execute(f"SELECT COALESCE(SUM(amount),0) AS total, COUNT(*) AS count FROM expenses{w}", p).fetchone()
            return {"total": round(r["total"], 2), "count": r["count"], "month": month or "all time", "category": category or "all"}

    @mcp.tool()
    def summarize_by_category(month: str = "") -> list[dict]:
        """Spending per category, biggest first. Optional month YYYY-MM."""
        w, p = where(month, "")
        with conn_ro() as c:
            rows = c.execute(
                f"SELECT category, ROUND(SUM(amount),2) AS total, COUNT(*) AS count FROM expenses{w} "
                "GROUP BY category ORDER BY total DESC", p)
            return [dict(r) for r in rows]

    mcp.run()  # stdio transport


if "--mcp" in sys.argv:
    run_mcp_server()
    sys.exit(0)

# ======================================================================================
# 2) STREAMLIT APP + ADK AGENTS
# ======================================================================================
import asyncio
import calendar
import json
import os
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types
from mcp import StdioServerParameters

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DB_PATH = Path(os.getenv("EXPENSE_DB", Path(__file__).resolve().parent / "expenses.db"))
APP_NAME = "expense_tracker"
DASHBOARD_HTML = Path(__file__).resolve().parent / "dashboard.html"

WRITE_TOOLS = ["add_expense", "delete_expense"]  # recorder only
READ_TOOLS = ["list_expenses", "get_total", "summarize_by_category"]  # analyst only (no write!)


def mcp_toolset(allowed: list[str]) -> McpToolset:
    """Launch this same file in --mcp mode and expose ONLY the allowed tools (least privilege)."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).resolve()), "--mcp"],
                env={"EXPENSE_DB": str(DB_PATH)},
            ),
            timeout=30,
        ),
        tool_filter=allowed,
    )


def build_coordinator(currency: str = "RM") -> Agent:
    today = date.today().isoformat()
    cats = ", ".join(CATEGORIES)

    recorder = Agent(
        name="expense_recorder",
        model=MODEL,
        description="Records or deletes expenses. Use when the user reports money they spent or asks to delete an entry.",
        instruction=(
            f"You record expenses. Today is {today}; currency is {currency}. "
            f"For EACH expense in the request call add_expense once, choosing the best category from: {cats}. "
            "Resolve relative dates (yesterday, last Friday) to YYYY-MM-DD. "
            "If asked to delete, call delete_expense with the id. "
            "Reply with a short confirmation listing what was saved, including ids."
        ),
        tools=[mcp_toolset(WRITE_TOOLS)],
    )
    analyst = Agent(
        name="expense_analyst",
        model=MODEL,
        description="Answers questions about past spending: totals, categories, trends, recent entries.",
        instruction=(
            f"You analyse spending. Today is {today}; currency is {currency}. ALWAYS use the tools, never guess numbers. "
            "Months are YYYY-MM (this month = " + today[:7] + "). "
            "Give short answers with amounts formatted like 'RM 12.50' using the given currency, "
            "and add one practical money-saving observation when the data supports it."
        ),
        tools=[mcp_toolset(READ_TOOLS)],
    )
    return Agent(
        name="expense_coordinator",
        model=MODEL,
        description="Routes expense messages to the right specialist.",
        instruction=(
            "You are a friendly personal expense assistant.\n"
            "- If the user says they spent / paid / bought something, or wants to delete an entry: call expense_recorder "
            "with the user's full message.\n"
            "- If the user asks about spending (totals, breakdowns, history, comparisons): call expense_analyst.\n"
            "- If a message does both, call both.\n"
            "- For greetings or unrelated chat, answer briefly yourself.\n"
            "Finish with one concise, friendly answer."
        ),
        tools=[AgentTool(agent=recorder), AgentTool(agent=analyst)],
    )


async def run_agent(agent, prompt, service, user_id, session_id, on_event=None) -> str:
    if await service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id) is None:
        await service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=service)
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    final = ""
    try:
        async for ev in runner.run_async(user_id=user_id, session_id=session_id, new_message=msg):
            if on_event:
                on_event(ev)
            if ev.is_final_response() and ev.content and ev.content.parts:
                final = "".join(p.text for p in ev.content.parts if getattr(p, "text", None)) or final
    finally:
        await runner.close()  # stops the MCP subprocesses
    return final


# ---------------------------------------------------------------- UI helpers
def db_df(sql: str, params=()) -> pd.DataFrame:
    """Read-only access for the dashboard (agents use MCP instead)."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True) as c:
        try:
            return pd.read_sql_query(sql, c, params=params)
        except Exception:  # table not created yet
            return pd.DataFrame()


@st.cache_resource
def session_service() -> InMemorySessionService:
    return InMemorySessionService()


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def build_payload(currency: str, budget: float) -> dict:
    """Live numbers for dashboard.html (this month only)."""
    today = date.today()
    month = today.strftime("%Y-%m")
    df = db_df("SELECT id, spent_on, category, amount, note FROM expenses WHERE substr(spent_on,1,7)=?", (month,))
    days = calendar.monthrange(today.year, today.month)[1]
    daily = [0.0] * days
    categories = []
    if not df.empty:
        for d, amt in df.groupby(df["spent_on"].str[8:10].astype(int))["amount"].sum().items():
            daily[d - 1] = round(float(amt), 2)
        g = df.groupby("category")["amount"].sum().sort_values(ascending=False)
        categories = [{"name": k, "total": round(float(v), 2)} for k, v in g.items()]
    recent = db_df("SELECT id, spent_on AS date, category, amount, note FROM expenses ORDER BY spent_on DESC, id DESC LIMIT 6")
    return {
        "currency": currency, "month_label": today.strftime("%B %Y"), "total": round(float(df["amount"].sum()), 2) if not df.empty else 0,
        "count": int(len(df)), "budget": float(budget), "categories": categories, "daily": daily,
        "recent": recent.to_dict("records"), "today_day": today.day,
    }


def render_dashboard(payload: dict) -> None:
    """Load dashboard.html and inject the data. (</ is escaped so user notes can never close the script tag.)"""
    if not DASHBOARD_HTML.exists():
        st.warning("dashboard.html not found - showing a basic view instead.")
        st.metric("Spent this month", f"{payload['currency']} {payload['total']:,.2f}")
        return
    data = json.dumps(payload).replace("<", "\\u003c")  # data can never break out of the <script> block
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    start = html.index("/*__DATA__*/")
    end = html.index(";", start)
    page = html[:start] + data + html[end:]
    if hasattr(st, "iframe"):  # newer Streamlit
        st.iframe(page, height=760)
    else:  # Streamlit < 1.5x
        components.html(page, height=760, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="AI Expense Tracker", page_icon="💸", layout="wide")
    if "uid" not in st.session_state:
        st.session_state.uid = f"u-{uuid.uuid4().hex[:8]}"
        st.session_state.sid = f"s-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []

    with st.sidebar:
        st.title("💸 AI Expense Tracker")
        st.caption("Google ADK · MCP · SQLite · Streamlit")
        key = st.text_input("Gemini API key", value=os.getenv("GOOGLE_API_KEY") or secret("GOOGLE_API_KEY"),
                            type="password", help="Free key: https://aistudio.google.com/apikey")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
        currency = st.text_input("Currency", value="RM")
        budget = st.number_input("Monthly budget (0 = off)", min_value=0.0, value=2000.0, step=100.0)
        st.markdown(f"**Model:** `{MODEL}`")
        st.markdown("**Try saying**\n- *Spent 25 on lunch and 80 on petrol*\n- *Paid 120 electricity bill yesterday*\n"
                    "- *How much did I spend on food this month?*\n- *Show my last 5 expenses*\n- *Delete expense 3*")
        if st.button("🗑️ Clear all data"):
            DB_PATH.unlink(missing_ok=True)
            st.session_state.messages = []
            st.rerun()

    # ---- dashboard (dashboard.html + live SQLite data)
    render_dashboard(build_payload(currency, budget))

    # ---- chat
    st.markdown("##### 💬 Ask your assistant")
    chips = ["Where does my money go this month?", "Show my last 5 expenses", "What was my biggest expense?", "Give me one saving tip"]
    quick = None
    for col, text in zip(st.columns(len(chips)), chips):
        if col.button(text, disabled=not key):
            quick = text
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                st.caption(m["trace"])

    if prompt := (st.chat_input("e.g. Spent 25 on lunch and 80 on petrol", disabled=not key) or quick):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        routed: list[str] = []
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                answer = asyncio.run(run_agent(
                    build_coordinator(currency), prompt, session_service(), st.session_state.uid,
                    st.session_state.sid, lambda ev: routed.extend(c.name for c in (ev.get_function_calls() or []))))
            except Exception as e:
                answer = f"⚠️ Something went wrong: `{e}`"
        trace = "🔀 Routed to: " + ", ".join(routed) if routed else ""
        st.session_state.messages.append({"role": "assistant", "content": answer or "(no answer)", "trace": trace})
        st.rerun()  # refresh dashboard with new data
    if not key:
        st.info("👈 Add your Gemini API key in the sidebar to start.")


if __name__ == "__main__":
    main()
