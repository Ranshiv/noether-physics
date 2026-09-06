"""Auditing a draft you are about to submit.

This is the workflow that pays for itself. arXiv now bans authors who submit
hallucinated references, and the audits behind that policy found a 17% phantom
rate in AI-assisted papers. Running this before you submit turns that risk into
a list you can act on.

It differs from :meth:`Library.audit_paper` in an important way: the draft is
*yours*, not something already in the corpus. Nothing is ingested, nothing is
stored as a paper, and the file is read where it sits. Only resolution results
are cached, because those are facts about the outside world rather than about
your document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .index.store import Store
from .ingest.bibliography import BibEntry
from .ingest.latex import LatexIngestError, parse_document
from .netclient import NetClient
from .verify.citations import AuditFinding, CitationResolver


@dataclass
class DraftAudit:
    """The result of auditing one draft."""

    path: Path
    title: str
    cite_keys: set[str] = field(default_factory=set)
    references: list[BibEntry] = field(default_factory=list)
    findings: list[AuditFinding] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def unresolvable(self) -> list[AuditFinding]:
        """The ones that would put a submission at risk."""
        return [f for f in self.findings if f.kind == "unresolvable"]

    @property
    def mismatches(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.kind == "mismatch"]

    @property
    def uncited(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.kind == "cited-unread"]

    @property
    def unchecked(self) -> list[AuditFinding]:
        """References nobody looked up -- offline mode, not a negative result."""
        return [f for f in self.findings if f.kind == "unchecked"]

    @property
    def resolved_count(self) -> int:
        """Only references actually confirmed. Unchecked ones are not resolved."""
        return len(self.references) - len(self.unresolvable) - len(self.unchecked)

    @property
    def undeclared_keys(self) -> set[str]:
        """Keys cited in the text with no matching bibliography entry.

        LaTeX renders these as a bold [?] and authors miss them routinely.
        """
        return self.cite_keys - {r.key for r in self.references}

    def summary(self) -> str:
        if self.error:
            return f"{self.path.name}: {self.error}"
        parts = [f"{len(self.references)} references, {self.resolved_count} resolved"]
        if self.unresolvable:
            parts.append(f"{len(self.unresolvable)} UNRESOLVABLE")
        if self.mismatches:
            parts.append(f"{len(self.mismatches)} mismatched")
        if self.undeclared_keys:
            parts.append(f"{len(self.undeclared_keys)} undeclared keys")
        return "; ".join(parts)

    @property
    def exit_code(self) -> int:
        """0 when the draft is safe to submit, 1 when it is not.

        Only unresolvable references and undeclared keys fail. A mismatch or an
        uncited entry is worth seeing but is not grounds for blocking a commit.
        """
        return 1 if (self.unresolvable or self.undeclared_keys) else 0


def audit_draft(
    path: Path,
    store: Store,
    net: NetClient | None = None,
    offline: bool = False,
) -> DraftAudit:
    """Audit a local ``.tex`` file's references without ingesting it.

    ``path`` may be the main file or the directory containing it.
    """
    path = Path(path)
    audit_error = DraftAudit(path=path, title="")
    if not path.exists():
        # Reading a nonexistent file yields an empty document that parses fine
        # and reports no problems -- the most misleading possible outcome.
        audit_error.error = "no such file or directory"
        return audit_error

    if path.is_dir():
        root, main = path, None
    else:
        root, main = path.parent, path

    audit = DraftAudit(path=path, title="")

    try:
        document, references = parse_document(
            root, paper_id=f"draft:{path.name}", main=main
        )
    except (LatexIngestError, OSError) as exc:
        audit.error = str(exc)
        return audit

    audit.title = document.title
    audit.cite_keys = document.cited_keys()
    audit.references = references

    resolver = CitationResolver(store, net or NetClient(), offline=offline)
    audit.findings = resolver.audit(references, cited_keys=audit.cite_keys)

    # A key cited with no bibliography entry is a distinct problem from an entry
    # that will not resolve, and LaTeX only signals it as a bold [?].
    for key in sorted(audit.undeclared_keys):
        audit.findings.append(
            AuditFinding(
                key=key,
                kind="undeclared",
                detail="cited in the text but absent from the bibliography",
            )
        )

    return audit


def render_audit(audit: DraftAudit) -> str:
    """A report a human can act on, ordered by severity."""
    if audit.error:
        return f"could not read {audit.path}: {audit.error}"

    lines = [f"{audit.path.name}  —  {audit.title[:70]}", ""]
    lines.append(f"{len(audit.cite_keys)} cite keys · {len(audit.references)} bibliography entries")
    lines.append(f"{audit.resolved_count}/{len(audit.references)} resolved")
    lines.append("")

    groups = [
        ("UNRESOLVABLE — no provider has such a record", audit.unresolvable),
        ("UNCHECKED — not looked up (offline); this is not a negative result",
         audit.unchecked),
        ("UNDECLARED — cited but not in the bibliography",
         [f for f in audit.findings if f.kind == "undeclared"]),
        ("MISMATCH — resolves, but to something else", audit.mismatches),
        ("UNCITED — listed but never cited in the text", audit.uncited),
        ("NO IDENTIFIER — matched by text only, treat as provisional",
         [f for f in audit.findings if f.kind == "no-identifier"]),
    ]

    for heading, findings in groups:
        if not findings:
            continue
        lines.append(heading)
        for finding in findings:
            lines.append(f"  {finding.key}")
            lines.append(f"    {finding.detail}")
        lines.append("")

    if not audit.findings:
        lines.append("no problems found")

    return "\n".join(lines)
