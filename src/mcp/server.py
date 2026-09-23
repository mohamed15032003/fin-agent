"""
Serveur MCP exposant les outils de l'agent (ratios, incohérences, résumé de
risques) de façon standardisée, pour qu'un client MCP (ex: Claude Desktop,
un autre agent) puisse les appeler directement.

C'est la partie différenciante du projet (semaine 4) — peu de projets
étudiants exposent leurs outils via MCP plutôt que juste un endpoint REST.

Doc officielle du protocole: https://modelcontextprotocol.io/

NB: le SDK `mcp` est passé en v2 en cours de route ; `FastMCP` (v1) a été
renommé `MCPServer`, déplacé vers `mcp.server.mcpserver`, mais garde la même
interface ergonomique (`.tool()`, `.run()`) — voir requirements.txt qui pin
la version installée.
"""

from mcp.server.mcpserver import MCPServer

from src.rag.retriever import retrieve
from src.rag.tools.ratios import compute_ratios
from src.rag.tools.inconsistency_detector import detect_inconsistencies
from src.rag.tools.risk_summarizer import summarize_risks

mcp = MCPServer("fin-agent")


@mcp.tool()
def get_relevant_context(query: str) -> str:
    """Récupère les passages de rapports financiers pertinents pour une requête."""
    hits = retrieve(query)
    return "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)


@mcp.tool()
def financial_ratios(query: str) -> str:
    """Calcule les ratios financiers pertinents pour la requête donnée."""
    return compute_ratios.invoke(query)


@mcp.tool()
def inconsistency_check(query: str) -> str:
    """Détecte des incohérences chiffrées dans les rapports liés à la requête."""
    return detect_inconsistencies.invoke(query)


@mcp.tool()
def risk_summary(query: str) -> str:
    """Résume les facteurs de risque liés à la requête donnée."""
    return summarize_risks.invoke(query)


if __name__ == "__main__":
    mcp.run()
