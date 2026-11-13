"""💸 AI Expense Tracker - Google ADK + MCP + SQLite + Streamlit (single file).

Architecture (same method as the Banking Virtual Analyst reference notebook):
  * Fine-grained MCP tools with RUNTIME DISCOVERY (list_tables -> describe_table -> read_query)
  * Protocol-enforced least privilege: tool_filter gives the analyst READ tools only, the recorder WRITE tools only
  * Defense in depth: before_model guardrail -> sqlparse AST check -> read-only SQLite mount -> PDPA masking
  * Live schema evolution demo: add tables mid-session, the agent adapts with ZERO code changes
  * This same file is also the MCP server:  `python app.py --mcp`
"""
import asyncio
import calendar
import json
import os
import re
import sqlite3
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import sqlparse

# ==============================================================================
# 1. Configuration
# ==============================================================================
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
APP_FILE = HERE / "app.py"
DASHBOARD_HTML = HERE / "dashboard.html"
DB_FILE = Path(os.getenv("EXPENSE_DB", HERE / "expenses.db"))
APP_NAME = "expense_tracker"
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Health", "Entertainment", "Other"]
MAX_ROWS = 50  # row cap for read_query results

# ==============================================================================
# 2. Database Bootstrap & Migration Functions
# ==============================================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS expenses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    amount     REAL NOT NULL,
    category   TEXT NOT NULL,
    note       TEXT,
    spent_on   TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def ensure_database(seed_demo: bool = True) -> None:
    """Create expenses.db if missing. A brand-new DB gets realistic sample data so the app works immediately."""
    is_new = not DB_FILE.exists()
    conn = sqlite3.connect(DB_FILE)
    conn.executescript(SCHEMA_SQL)
    if is_new and seed_demo:
        today = date.today()
        samples = [
            (12.50, "Food", "Nasi lemak breakfast"), (45.00, "Transport", "Grab to office"),
            (89.90, "Shopping", "New shirt"), (150.00, "Bills", "Internet bill"),
            (32.00, "Food", "Lunch with team"), (60.00, "Health", "Pharmacy"),
            (28.00, "Entertainment", "Movie ticket"), (18.50, "Food", "Coffee and cake"),
            (70.00, "Transport", "Petrol"),
        ]
        for i, (amount, category, note) in enumerate(samples):  # spread backwards from today, inside this month
            day = date(today.year, today.month, max(1, today.day - i))
            conn.execute("INSERT INTO expenses(amount, category, note, spent_on) VALUES (?,?,?,?)",
                         (amount, category, note, day.isoformat()))
    conn.commit()
    conn.close()


def execute_demo_migration() -> None:
    """Simulate a live schema evolution: add 2 NEW tables. No agent code or prompt is touched."""
    conn = sqlite3.connect(DB_FILE)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS budgets (
        category      TEXT PRIMARY KEY,
        monthly_limit REAL NOT NULL
    );
    INSERT OR IGNORE INTO budgets VALUES
        ('Food', 400), ('Transport', 150), ('Shopping', 200), ('Bills', 300),
        ('Health', 100), ('Entertainment', 80), ('Other', 100);

    CREATE TABLE IF NOT EXISTS savings_goals (
        goal_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL UNIQUE,
        target_amount REAL NOT NULL,
        saved_amount  REAL NOT NULL,
        deadline      TEXT
    );
    INSERT OR IGNORE INTO savings_goals(name, target_amount, saved_amount, deadline) VALUES
        ('New laptop', 4500, 1800, '2027-03-31'),
        ('Emergency fund', 10000, 3200, '2027-12-31'),
        ('Family holiday', 3000, 2100, '2027-06-30');
    """)
    conn.commit()
    conn.close()


def get_live_ddl() -> str:
    """Read the CURRENT DDL straight from the live database (always up to date)."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    conn.close()
    return "\n\n".join(r[0] for r in rows if r[0])


# ==============================================================================
# 3. Security Primitives (pure Python - easy to test)
# ==============================================================================
FORBIDDEN_SQL = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "CREATE",
    "RENAME", "ATTACH", "DETACH", "PRAGMA", "UPSERT", "VACUUM", "GRANT", "REVOKE",
}


def is_sql_mutation(sql_query: str) -> bool:
    """AST-based check: only ONE plain SELECT statement is allowed."""
    statements = [s for s in sqlparse.parse(sql_query.strip().strip(";")) if str(s).strip()]
    if not statements:
        return False
    if len(statements) > 1:  # stacked queries such as "SELECT 1; DROP TABLE x"
        return True
    for stmt in statements:
        if stmt.get_type() != "SELECT":
            return True
        for token in stmt.flatten():
            if token.value.upper() in FORBIDDEN_SQL:
                return True
    return False


DESTRUCTIVE_INTENTS = [
    r"\bdrop\s+(table|database)\b", r"\btruncate\b", r"\bwipe\b", r"\bdelete\s+from\b",
    r"\balter\s+table\b", r"\bupdate\s+\w+\s+set\b",
    r"\bdelete\s+(all|every|everything|the\s+(whole\s+)?(table|database|data))\b",
    r"\b(erase|clear|remove|reset)\s+(all|every|everything)\b",
    r"\bignore\s+(all\s+|your\s+)?(previous|prior)\s+instructions\b",
]


def is_destructive_intent(text: str) -> bool:
    """Bulk / destructive requests are blocked BEFORE the LLM sees them. (Deleting ONE entry by id is allowed.)"""
    return any(re.search(p, text.lower()) for p in DESTRUCTIVE_INTENTS)


def _mask_digits(text: str, keep_last: int) -> str:
    """Replace every digit with * except the last `keep_last` digits (separators are kept)."""
    total, seen, out = sum(c.isdigit() for c in text), 0, []
    for c in text:
        if c.isdigit():
            seen += 1
            out.append(c if seen > total - keep_last else "*")
        else:
            out.append(c)
    return "".join(out)


def anonymize_text(text: str) -> str:
    """PDPA masking: emails, card numbers, Malaysian NRIC and phone numbers."""
    text = re.sub(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", r"\1***@\2", text)
    text = re.sub(r"\b(?:\d[ -]?){12,18}\d\b", lambda m: _mask_digits(m.group(), 4), text)           # card 13-19 digits
    text = re.sub(r"\b\d{6}-?\d{2}-?\d{4}\b", lambda m: _mask_digits(m.group(), 4), text)           # NRIC 12 digits
    text = re.sub(r"(?<![\w+])(?:\+?60[- ]?|0)1\d[- ]?\d{3,4}[- ]?\d{4}(?!\d)",
                  lambda m: _mask_digits(m.group(), 3), text)                                         # MY mobile
    return text


# ==============================================================================
# 4. MCP Database Server  (run with: python app.py --mcp)
# ==============================================================================
def run_mcp_server() -> None:
    """Real MCP server over stdio. The agents discover these tools at runtime via McpToolset."""
    try:  # mcp>=2 renamed FastMCP -> MCPServer
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server

    mcp = Server("expense-sqlite")
    ensure_database(seed_demo=False)

    def connect_ro() -> sqlite3.Connection:  # OS-level read-only mount
        conn = sqlite3.connect(f"{DB_FILE.resolve().as_uri()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        return conn

    # ---------------- READ tools (analyst) ----------------
    @mcp.tool()
    def mcp_list_tables() -> dict:
        """Discover all data tables that currently exist in the expense database."""
        conn = connect_ro()
        try:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
            return {"status": "success", "tables": [r[0] for r in rows]}
        finally:
            conn.close()

    @mcp.tool()
    def mcp_describe_table(table_name: str) -> dict:
        """Get column names, types and primary keys (and row count) for one table, e.g. 'expenses'."""
        name = table_name.strip().strip("'\"` ")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):  # identifier whitelist (no SQL injection)
            return {"status": "error", "message": "Invalid table name."}
        conn = connect_ro()
        try:
            cols = [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]}
                    for r in conn.execute(f'PRAGMA table_info("{name}")')]
            if not cols:
                return {"status": "error", "message": f"Table '{name}' not found."}
            count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            return {"status": "success", "table": name, "columns": cols, "row_count": count}
        finally:
            conn.close()

    @mcp.tool()
    def mcp_read_query(sql_query: str) -> dict:
        """Run ONE read-only SQLite SELECT statement (max 50 rows). Use strftime/substr on the spent_on column
        (format YYYY-MM-DD) for month filters, e.g. substr(spent_on,1,7) = '2026-11'."""
        if is_sql_mutation(sql_query):
            return {"status": "blocked",
                    "error": "🛡️ [MCP PROTOCOL BLOCKED]: Only a single SELECT is allowed. Mutations are not supported by this endpoint."}
        conn = connect_ro()
        try:
            cur = conn.execute(sql_query.strip())
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(MAX_ROWS + 1)
            return {"status": "success", "columns": columns, "rows": [list(r) for r in rows[:MAX_ROWS]],
                    "row_count": min(len(rows), MAX_ROWS), "truncated": len(rows) > MAX_ROWS}
        except sqlite3.Error as e:
            return {"status": "error", "message": str(e)}
        finally:
            conn.close()

    # ---------------- WRITE tools (recorder) ----------------
    @mcp.tool()
    def mcp_add_expense(amount: float, category: str, note: str = "", spent_on: str = "") -> dict:
        """Record ONE expense. category: Food, Transport, Shopping, Bills, Health, Entertainment or Other.
        spent_on is YYYY-MM-DD (empty = today)."""
        if amount <= 0:
            return {"status": "error", "message": "amount must be greater than 0"}
        cat = next((c for c in CATEGORIES if c.lower() == category.strip().lower()), "Other")
        try:
            day = date.fromisoformat(spent_on) if spent_on else date.today()
        except ValueError:
            return {"status": "error", "message": "spent_on must be YYYY-MM-DD"}
        safe_note = anonymize_text(note)[:200]  # PDPA: never store raw card / NRIC / phone numbers
        conn = sqlite3.connect(DB_FILE)
        try:
            cur = conn.execute("INSERT INTO expenses(amount, category, note, spent_on) VALUES (?,?,?,?)",
                               (round(amount, 2), cat, safe_note, day.isoformat()))
            conn.commit()
            return {"status": "success", "id": cur.lastrowid, "amount": round(amount, 2), "category": cat,
                    "note": safe_note, "spent_on": day.isoformat()}
        finally:
            conn.close()

    @mcp.tool()
    def mcp_delete_expense(expense_id: int) -> dict:
        """Delete ONE expense by its id. (Bulk deletion is intentionally not available.)"""
        conn = sqlite3.connect(DB_FILE)
        try:
            n = conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,)).rowcount
            conn.commit()
            return {"status": "success" if n else "error", "deleted": bool(n), "id": expense_id}
        finally:
            conn.close()

    mcp.run()  # stdio transport


if "--mcp" in sys.argv:  # MCP subprocess mode: must run BEFORE Streamlit is imported
    run_mcp_server()
    sys.exit(0)

# ==============================================================================
# 5. Streamlit + Google ADK imports, API key resolution
# ==============================================================================
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google.adk.agents import Agent
from google.adk.models import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types
from mcp import StdioServerParameters

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


def resolve_api_key() -> str:
    """Safe secret resolution: Streamlit Cloud secrets first, then environment variable."""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")


# ==============================================================================
# 6. Guardrail, MCP Toolsets (least privilege) and Agents
# ==============================================================================
READ_TOOLS = ["mcp_list_tables", "mcp_describe_table", "mcp_read_query"]  # analyst: can NEVER write
WRITE_TOOLS = ["mcp_add_expense", "mcp_delete_expense"]  # recorder: can write, cannot run free SQL

# Auto-retry when Gemini is overloaded (503) or rate-limited (429): waits ~1s, 2s, 4s, 8s ...
RETRY_CONFIG = types.GenerateContentConfig(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=5, initial_delay=1.0, max_delay=20.0,
                                             http_status_codes=[429, 500, 503, 504])
    )
)


def before_model_guardrail(callback_context, llm_request):
    """Short-circuits destructive / injection prompts BEFORE the LLM is called (zero tokens spent)."""
    last_text = ""
    if llm_request.contents and llm_request.contents[-1].parts:
        last_text = llm_request.contents[-1].parts[0].text or ""
    if last_text and is_destructive_intent(last_text):
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(
            text="🛡️ **[GUARDRAIL INTERVENTION]**: Blocked by the Expense Safety Policy. Bulk deletion or data "
                 "modification is not permitted. You can delete a single entry by its id, e.g. *Delete expense 3*.")]))
    return None


def mcp_toolset(allowed: list[str]) -> McpToolset:
    """Launch THIS file in --mcp mode and expose ONLY the allowed tools (protocol-level least privilege)."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable, args=[str(APP_FILE), "--mcp"], env={"EXPENSE_DB": str(DB_FILE)}),
            timeout=30,
        ),
        tool_filter=allowed,
    )


def build_coordinator(currency: str = "RM", model: str = MODEL) -> Agent:
    """Coordinator (router) -> expense_recorder (write) | expense_analyst (read-only). Agent-as-Tool pattern."""
    today = date.today().isoformat()
    cats = ", ".join(CATEGORIES)

    recorder = Agent(
        name="expense_recorder",
        model=model,
        generate_content_config=RETRY_CONFIG,
        description="Records or deletes ONE expense at a time. Use when the user reports money they spent or asks to delete an entry by id.",
        instruction=(
            f"You record expenses. Today is {today}; currency is {currency}.\n"
            f"For EACH expense in the request call mcp_add_expense once, choosing the best category from: {cats}.\n"
            "Resolve relative dates (yesterday, last Friday) to YYYY-MM-DD.\n"
            "If asked to delete ONE entry, call mcp_delete_expense with its id. Never attempt bulk deletion.\n"
            "Reply with a short confirmation listing what was saved, including ids."
        ),
        tools=[mcp_toolset(WRITE_TOOLS)],
        before_model_callback=before_model_guardrail,
    )
    analyst = Agent(
        name="expense_analyst",
        model=model,
        generate_content_config=RETRY_CONFIG,
        description="Answers questions about spending data: totals, categories, trends, budgets, goals, history.",
        instruction=(
            f"You are a personal finance Data Analyst operating through the Model Context Protocol (MCP). Today is {today}; "
            f"currency is {currency}.\n"
            "You do NOT have a static hardcoded database schema. Interact with the database dynamically:\n"
            "1. Call 'mcp_list_tables' to see which tables currently exist.\n"
            "2. Call 'mcp_describe_table' to inspect the columns of any table you need.\n"
            "3. Call 'mcp_read_query' to run read-only SELECT statements.\n\n"
            "RULES:\n"
            "- Always verify table and column names with the MCP tools instead of assuming. New tables can appear at any time "
            "(for example budgets or savings goals) - use them when they help answer the question.\n"
            "- Dates are stored as text YYYY-MM-DD in the spent_on column; this month = " + today[:7] + ".\n"
            "- Never invent numbers. Format money like 'RM 12.50' using the given currency.\n"
            "- Summarize findings into short, practical insights.\n"
            "- Mutation operations are strictly prohibited."
        ),
        tools=[mcp_toolset(READ_TOOLS)],
        before_model_callback=before_model_guardrail,
    )
    return Agent(
        name="expense_coordinator",
        model=model,
        generate_content_config=RETRY_CONFIG,
        description="Routes expense messages to the right specialist.",
        instruction=(
            "You are a friendly personal expense assistant.\n"
            "- If the user says they spent / paid / bought something, or wants to delete one entry: call expense_recorder "
            "with the user's full message.\n"
            "- If the user asks about spending, tables, budgets, goals, history or comparisons: call expense_analyst.\n"
            "- If a message does both, call both.\n"
            "- For greetings or 'Who are you?', introduce yourself briefly as the MCP Expense Tracker assistant.\n"
            "Finish with one concise, friendly answer."
        ),
        tools=[AgentTool(agent=recorder), AgentTool(agent=analyst)],
        before_model_callback=before_model_guardrail,
    )


# ==============================================================================
# 7. Async Runner (fresh agents per request; closes the MCP subprocesses afterwards)
# ==============================================================================
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
        await runner.close()  # stops the MCP stdio subprocesses
    return final


# ==============================================================================
# 8. Dashboard helpers (dashboard.html + live SQLite data)
# ==============================================================================
def db_df(sql: str, params=()) -> pd.DataFrame:
    """Read-only access for the dashboard (agents use MCP instead)."""
    if not DB_FILE.exists():
        return pd.DataFrame()
    with sqlite3.connect(f"{DB_FILE.resolve().as_uri()}?mode=ro", uri=True) as c:
        try:
            return pd.read_sql_query(sql, c, params=params)
        except Exception:
            return pd.DataFrame()


def build_payload(currency: str, budget: float) -> dict:
    """Live numbers for dashboard.html (this month only)."""
    today = date.today()
    df = db_df("SELECT id, spent_on, category, amount, note FROM expenses WHERE substr(spent_on,1,7)=?",
               (today.strftime("%Y-%m"),))
    daily = [0.0] * calendar.monthrange(today.year, today.month)[1]
    categories = []
    if not df.empty:
        for d, amt in df.groupby(df["spent_on"].str[8:10].astype(int))["amount"].sum().items():
            daily[d - 1] = round(float(amt), 2)
        g = df.groupby("category")["amount"].sum().sort_values(ascending=False)
        categories = [{"name": k, "total": round(float(v), 2)} for k, v in g.items()]
    recent = db_df("SELECT id, spent_on AS date, category, amount, note FROM expenses ORDER BY spent_on DESC, id DESC LIMIT 6")
    if not recent.empty:
        recent["note"] = recent["note"].fillna("").map(anonymize_text)  # PDPA masking on display too
    return {
        "currency": currency, "month_label": today.strftime("%B %Y"),
        "total": round(float(df["amount"].sum()), 2) if not df.empty else 0, "count": int(len(df)),
        "budget": float(budget), "categories": categories, "daily": daily,
        "recent": recent.to_dict("records"), "today_day": today.day,
    }


def render_dashboard(payload: dict) -> None:
    """Load dashboard.html and inject the data (every '<' is escaped so data can never break out of <script>)."""
    if not DASHBOARD_HTML.exists():
        st.warning("dashboard.html not found - showing a basic view instead.")
        st.metric("Spent this month", f"{payload['currency']} {payload['total']:,.2f}")
        return
    data = json.dumps(payload).replace("<", "\\u003c")
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    start = html.index("/*__DATA__*/")
    end = html.index(";", start)
    page = html[:start] + data + html[end:]
    if hasattr(st, "iframe"):  # newer Streamlit
        st.iframe(page, height=760)
    else:  # Streamlit < 1.5x
        components.html(page, height=760, scrolling=True)


@st.cache_resource
def get_session_service() -> InMemorySessionService:
    return InMemorySessionService()


# ==============================================================================
# 9. Streamlit App: Sidebar (demo + suggested questions) -> Dashboard -> Chat
# ==============================================================================
SAMPLE_QUESTIONS = [
    "What tables currently exist in my database?",
    "How much did I spend on food this month?",
    "Show my last 5 expenses.",
    "Spent 25 on lunch and 80 on petrol",
    "Which categories are over budget this month?",
    "How close am I to my savings goals?",
    "Who are you?",
    "delete all my expenses",
]


def main() -> None:
    st.set_page_config(page_title="AI Expense Tracker (MCP)", page_icon="💸", layout="wide",
                       initial_sidebar_state="expanded")
    ensure_database()
    if "uid" not in st.session_state:
        st.session_state.uid = f"u-{uuid.uuid4().hex[:8]}"
        st.session_state.sid = f"s-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []

    with st.sidebar:
        st.title("💸 Expense Tracker")
        st.markdown("**AI-Powered Personal Finance (MCP Architecture)**")
        key = st.text_input("Gemini API key", value=resolve_api_key(), type="password",
                            help="Free key: https://aistudio.google.com/apikey")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
        currency = st.text_input("Currency", value="RM")
        budget = st.number_input("Monthly budget (0 = off)", min_value=0.0, value=2000.0, step=100.0)
        model = st.text_input("Model", value=MODEL, help="If you see 503 errors, try another Gemini model name here.")
        st.markdown("---")

        # ---- live database evolution simulator (MCP demo)
        st.subheader("⚡ Live Database Evolution (MCP Demo)")
        st.caption("Add new tables mid-session. The MCP agent discovers them at runtime with zero code changes.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Add Tables", help="Migrates `budgets` and `savings_goals` into SQLite"):
                execute_demo_migration()
                st.toast("Added `budgets` and `savings_goals` tables!")
                st.rerun()
        with col2:
            if st.button("🔄 Reset DB", help="Back to the sample expenses only"):
                DB_FILE.unlink(missing_ok=True)
                ensure_database(seed_demo=True)
                st.toast("Database reset to sample data.")
                st.rerun()
        if st.button("🧹 Clear all data (empty DB)"):
            DB_FILE.unlink(missing_ok=True)
            ensure_database(seed_demo=False)
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")

        # ---- suggested questions
        st.subheader("💡 Suggested Questions")
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if st.button(q, key=f"sq{i}", disabled=not key):
                st.session_state.pending_prompt = q
        st.markdown("---")
        with st.expander("📄 View Live Database Schema", expanded=False):
            st.text(get_live_ddl())
        st.caption("🔒 **Security:** MCP tool_filter least privilege | AST read-only guard | Read-only mount | PDPA masking")

    st.title("💸 AI Expense Tracker")
    st.caption("Connected via Model Context Protocol | Zero-Touch Schema Evolution | PDPA Masking Active")
    render_dashboard(build_payload(currency, budget))

    # ---- chat history
    st.markdown("##### 💬 Ask your assistant")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                st.caption(m["trace"])

    # ---- prompt resolution (sidebar button vs chat box)
    prompt_to_send = None
    if st.session_state.get("pending_prompt"):
        prompt_to_send = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        chat_input = st.chat_input("e.g. Spent 25 on lunch and 80 on petrol", disabled=not key)
        if chat_input:
            prompt_to_send = chat_input
    if not key:
        st.info("👈 Add your Gemini API key in the sidebar to start.")

    # ---- run the agent workflow
    if prompt_to_send:
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)
        routed: list[str] = []
        with st.chat_message("assistant"), st.spinner("Executing MCP data tools and verifying guardrails..."):
            try:
                raw = asyncio.run(run_agent(
                    build_coordinator(currency, model), prompt_to_send, get_session_service(), st.session_state.uid,
                    st.session_state.sid, lambda ev: routed.extend(c.name for c in (ev.get_function_calls() or []))))
            except Exception as e:
                raw = f"⚠️ Something went wrong: `{e}`"
        reply = anonymize_text(raw or "No response generated.")  # PDPA masking on the way out
        trace = "🔀 Routed to: " + ", ".join(routed) if routed else ""
        st.session_state.messages.append({"role": "assistant", "content": reply, "trace": trace})
        st.rerun()  # refresh dashboard with the new data


if __name__ == "__main__":
    main()
