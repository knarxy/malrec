from __future__ import annotations

import json
import logging

import typer

from . import feedback as fb
from .config import settings
from .db import migrate, one, query, refresh_franchises, scalar
from .eval import compare_algos, kfold, temporal_holdout, tune_implicit, tune_recency
from .ingest import jobs
from .model import feature_importance
from .model import train as train_model
from .surfaces import SURFACES, build_all, build_surface

app = typer.Typer(add_completion=False, help="MyAnimeList recommendation engine")
sync_app = typer.Typer(help="Data ingestion")
app.add_typer(sync_app, name="sync")


def _setup(verbose: bool = True) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


def _uid(user: str | None) -> int:
    name = user or settings().malrec_user
    row = one("SELECT id FROM app_user WHERE mal_username=%s", (name,))
    if not row:
        typer.echo(f"user {name!r} not synced. Run: malrec sync list", err=True)
        raise typer.Exit(1)
    return row["id"]


@app.command()
def init() -> None:
    """Apply database migrations."""
    _setup()
    for name in migrate():
        typer.echo(f"  applied {name}")
    from .tokenbox import reseal_all
    n = reseal_all()
    if n:
        typer.echo(f"  encrypted {n} stored MAL session(s)")
    typer.echo("database ready")


@sync_app.command("list")
def sync_list(user: str = typer.Option(None)) -> None:
    """Pull the user's MAL list (one request; no OAuth needed for public lists)."""
    _setup()
    typer.echo(json.dumps(jobs.sync_user_list(user), indent=2))


@sync_app.command("catalog")
def sync_catalog() -> None:
    """Pull the candidate universe from the ranking and seasonal endpoints."""
    _setup()
    typer.echo(json.dumps(jobs.sync_catalog(), indent=2))


@sync_app.command("graph")
def sync_graph(limit: int = typer.Option(None, help="cap the number of anime fetched"),
               scope: str = typer.Option("listed", help="'listed' (your list only) or 'all'"),
               all_: bool = typer.Option(False, "--all", help="refetch, ignoring the marker")) -> None:
    """Pull MAL recommendation edges for your list (~250 requests, ~3 min).

    scope='all' walks the whole catalogue instead; that costs ~45 minutes and
    is only worth it for features that need candidate-to-candidate edges.
    """
    _setup()
    typer.echo(json.dumps(jobs.sync_graph(limit, only_missing=not all_, scope=scope), indent=2))


@sync_app.command("anilist")
def sync_anilist(limit: int = typer.Option(None),
                 all_: bool = typer.Option(False, "--all")) -> None:
    """Pull AniList weighted tags, scores and its recommendation graph."""
    _setup()
    typer.echo(json.dumps(jobs.sync_anilist(limit, only_missing=not all_), indent=2))


@sync_app.command("all")
def sync_all(user: str = typer.Option(None), graph_limit: int = typer.Option(None)) -> None:
    """Cold start: list, catalog, graph, AniList. Takes ~25 minutes."""
    _setup()
    typer.echo(json.dumps(jobs.bootstrap(user, graph_limit), indent=2))


@sync_app.command("full")
def sync_full(part: str = typer.Option("all", help="'mal', 'anilist', 'all' or 'backfill'")) -> None:
    """One-time catalogue-wide fetch (MAL + AniList + CF sample). Resumable;
    every item is stored on arrival and never requested twice."""
    from .ingest.fullfetch import run
    _setup()
    typer.echo(json.dumps(run(part), indent=2, default=str))


@sync_app.command("users")
def sync_users_cmd() -> None:
    """Re-read every app user's MAL list; rebuild those whose list changed."""
    _setup()
    from .refresh import sync_all_users
    typer.echo(json.dumps(sync_all_users(), indent=2, default=str))


@sync_app.command("upcoming")
def sync_upcoming_cmd(rebuild: bool = typer.Option(True, help="rebuild every user's coming_soon")) -> None:
    """Catalogue + relations of newly announced titles (~30-60 requests)."""
    _setup()
    from .ingest.jobs import sync_upcoming
    out = sync_upcoming()
    if rebuild:
        from .surfaces import build_surface
        failed = {}
        for r in query("SELECT id, mal_username FROM app_user ORDER BY id"):
            try:
                build_surface(r["id"], "coming_soon", limit=60)
            except Exception as e:  # noqa: BLE001 - one broken account must not skip the rest
                logging.getLogger(__name__).warning("coming_soon for %s failed: %s",
                                                    r["mal_username"], e)
                failed[r["mal_username"]] = f"{type(e).__name__}: {e}"[:200]
        out["rebuilt"] = "coming_soon for every user" + (" except those failed" if failed else "")
        if failed:
            out["failed"] = failed
    typer.echo(json.dumps(out, indent=2, default=str))


@sync_app.command("cf-refresh")
def sync_cf_refresh(new_users: int = typer.Option(800),
                    retire_days: int = typer.Option(365)) -> None:
    """Rotate the population sample: add new users from recent discussion
    replies, retire as many lists older than `retire_days`. Then run
    `malrec population fit`. Meant to run about monthly."""
    _setup()
    from .clients.mal import MalClient
    from .ingest.fullfetch import cf_refresh
    with MalClient() as mal:
        typer.echo(json.dumps(cf_refresh(mal, new_users, retire_days), indent=2, default=str))


@app.command("report")
def report_cmd() -> None:
    """Gate vs baselines, range coverage and the prospective test, per user."""
    _setup(verbose=False)
    from .report import report
    typer.echo(json.dumps(report(), indent=2, default=str))


@app.command("eval-prospective")
def eval_prospective_cmd(user: str = typer.Option(None)) -> None:
    """Shown predictions vs scores given afterwards - the out-of-sample test."""
    _setup(verbose=False)
    from .eval import prospective
    typer.echo(json.dumps(prospective(_uid(user)), indent=2, default=str))


@app.command("compare-prospective")
def compare_prospective_cmd(candidate: int = typer.Option(
        None, help="population model id to compare with the active one (default: itself)"),
        gate_user: str = typer.Option(None, help="must not get worse on its own (default: MALREC_USER)")) -> None:
    """Active vs candidate population model on the ratings given after the app
    recommended those titles - the check a model change must pass."""
    _setup(verbose=False)
    from .recsys.service import active_global, load_global, prospective_compare
    current = active_global()
    other = load_global(candidate) if candidate else current
    typer.echo(json.dumps(prospective_compare(current, other, gate_user or settings().malrec_user),
                          indent=2, default=str))


@app.command("learn-weights")
def learn_weights_cmd(min_events: int = typer.Option(200)) -> None:
    """What in-app actions say the relevance and novelty weights should be."""
    _setup()
    from .learn import learn_weights
    typer.echo(json.dumps(learn_weights(min_events), indent=2, default=str))


@sync_app.command("franchises")
def sync_franchises() -> None:
    """Recompute franchise connected components."""
    _setup()
    typer.echo(f"converged in {refresh_franchises()} passes")


population_app = typer.Typer(help="The population model learned from sampled MAL lists")
app.add_typer(population_app, name="population")


@population_app.command("fit")
def population_fit(gate_user: str = typer.Option(
        None, help="activate only if this user's temporal holdout does not get worse"),
        dry_run: bool = typer.Option(False, help="report the gate decision, activate nothing")
        ) -> None:
    """Fit item biases, co-rating similarities and latent factors from the CF
    sample, plus the stacker that blends them, and make it the active model
    (with --gate-user, only if it passes that user's holdout)."""
    from .recsys.service import fit_global, gated_fit
    _setup()
    out = (gated_fit(gate_user, dry_run=dry_run) if gate_user
           else fit_global(activate=not dry_run))
    typer.echo(json.dumps(out, indent=2, default=str))


@population_app.command("status")
def population_status() -> None:
    _setup(verbose=False)
    for r in query("SELECT id, active, created_at, meta FROM global_model ORDER BY id DESC LIMIT 5"):
        m = r["meta"]
        typer.echo(f"#{r['id']}{' (active)' if r['active'] else ''}  {r['created_at']:%Y-%m-%d %H:%M}"
                   f"  users={m.get('users')} items={m.get('items')} ratings={m.get('ratings')}")


@app.command()
def train(user: str = typer.Option(None), algo: str = typer.Option("ridge")) -> None:
    """Fit the taste model and store it."""
    _setup()
    model, run_id = train_model(_uid(user), algo=algo)
    typer.echo(f"\nmodel run {run_id} ({algo})")
    typer.echo(json.dumps(model.metrics, indent=2))
    typer.echo("\ntop learned weights:")
    for name, val in feature_importance(model, 20):
        typer.echo(f"  {val:+7.3f}  {name}")


@app.command()
def evaluate(user: str = typer.Option(None),
             method: str = typer.Option("temporal", help="temporal | kfold | both")) -> None:
    """Score the model. `temporal` is the one to trust for a recommender."""
    _setup(verbose=False)
    uid = _uid(user)
    if method in ("temporal", "both"):
        typer.echo("temporal holdout (train on older ratings, predict newest):")
        typer.echo(json.dumps(temporal_holdout(uid), indent=2))
    if method in ("kfold", "both"):
        typer.echo("\nrandom k-fold (do NOT use to judge recency weighting):")
        typer.echo(json.dumps(kfold(uid), indent=2))


@app.command("tune-recency")
def tune_recency_cmd(user: str = typer.Option(None)) -> None:
    """Grid-search the recency half-life on the temporal holdout."""
    _setup(verbose=False)
    rows = tune_recency(_uid(user))
    typer.echo(f"{'half-life':>10} {'floor':>6} {'spearman':>9} {'rmse':>7} {'ndcg@10':>8}")
    for r in rows:
        hl = "off" if r["half_life"] is None else f"{r['half_life']:g}y"
        typer.echo(f"{hl:>10} {r['floor']:>6.2f} {r['spearman'] or 0:9.3f} "
                   f"{r['rmse'] or 0:7.3f} {r['ndcg@10'] or 0:8.3f}")
    best = rows[0]
    typer.echo(f"\nbest: half_life={best['half_life']} floor={best['floor']}  "
               f"-> set recency_half_life_years / recency_floor in config.py")


@app.command("tune-implicit")
def tune_implicit_cmd(user: str = typer.Option(None)) -> None:
    """Fit the pseudo-rating offsets for unscored list entries."""
    _setup(verbose=False)
    rows = tune_implicit(_uid(user))
    typer.echo(f"{'weight':>7} {'dropped':>8} {'on_hold':>8} {'spearman':>9} "
               f"{'rmse':>7} {'ndcg@10':>8}  wins")
    for r in rows:
        off = r["offsets"]
        typer.echo(f"{r['weight']:>7.2f} {off.get('dropped', 0):>8.1f} "
                   f"{off.get('on_hold', 0):>8.1f} {r['spearman'] or 0:9.3f} "
                   f"{r['rmse'] or 0:7.3f} {r['ndcg@10'] or 0:8.3f}  {r['wins'] or '-'}")
    best = rows[0]
    typer.echo(f"\nbest: implicit_weight={best['weight']} offsets={best['offsets']}")


@app.command("compare-algos")
def compare_algos_cmd(user: str = typer.Option(None)) -> None:
    """Ridge vs LightGBM on the temporal holdout."""
    _setup(verbose=False)
    typer.echo(json.dumps(compare_algos(_uid(user)), indent=2, default=str))


@app.command()
def recommend(user: str = typer.Option(None),
              surface: str = typer.Option("discover", help=f"one of {list(SURFACES)}"),
              limit: int = typer.Option(25),
              rebuild: bool = typer.Option(True, help="recompute instead of reading the cache"),
              verbose: bool = typer.Option(False, "-v")) -> None:
    """Produce recommendations and print them."""
    _setup(verbose)
    uid = _uid(user)
    cands = (build_surface(uid, surface, limit=limit) if rebuild
             else [type("C", (), r)() for r in query(
                 "SELECT * FROM recommendation WHERE user_id=%s AND surface=%s ORDER BY rank",
                 (uid, surface))])
    typer.echo(f"\n{surface}  ({len(cands)} results)\n")
    typer.echo(f"{'#':>3} {'pred':>5} {'MAL':>5} {'pop':>6}  {'title':<44}  why")
    typer.echo("-" * 120)
    for i, c in enumerate(cands, 1):
        row = c.row or {}
        why = []
        for r in c.reasons[:2]:
            if r["kind"] in ("because_you_liked", "continues") and r.get("title"):
                why.append(f"{r['title'][:24]}({r.get('your_score')})")
            elif r["kind"] == "tags":
                why.append(", ".join(r["tags"][:3]))
        typer.echo(f"{i:>3} {c.predicted:5.2f} {row.get('mal_mean') or 0:5.2f} "
                   f"{row.get('mal_popularity') or 0:6}  {c.title[:44]:<44}  "
                   f"{' | '.join(why)[:52]}")


notify_app = typer.Typer(help="Admin e-mails (malrec.notify).")
app.add_typer(notify_app, name="notify")


def _stdin_json():
    """The first JSON value on stdin (a job's output), or None. Anything
    around it, or a broken document, never stops the mail from going out."""
    import sys
    raw = sys.stdin.read()
    start = min((i for i in (raw.find("{"), raw.find("[")) if i >= 0), default=-1)
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(raw[start:])[0]
    except ValueError:
        logging.getLogger(__name__).warning("notify: could not read the job's JSON output")
        return None


@notify_app.command("test")
def notify_test(all_kinds: bool = typer.Option(False, "--all", help="also one sample of every kind")
                ) -> None:
    """Send a test mail to ADMIN_EMAIL."""
    _setup()
    from .notify import enabled, test
    if not enabled():
        raise typer.Exit("SMTP_HOST / ADMIN_EMAIL not set")
    typer.echo(json.dumps(test(all_kinds)))


@notify_app.command("monthly")
def notify_monthly(log: str = typer.Option(..., help="the refresh log"),
                   dry_run: bool = typer.Option(False)) -> None:
    """Mail the monthly refresh's result; the fit JSON comes on stdin."""
    _setup()
    from pathlib import Path

    from .notify import monthly
    fit = _stdin_json() or {}
    text = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    start = text.rfind("monthly refresh")            # this run's section of the log
    monthly(fit, text[max(start, 0):], dry_run=dry_run)


@notify_app.command("weekly")
def notify_weekly() -> None:
    """Mail the weekly coming-soon refresh's result (its JSON on stdin)."""
    _setup()
    from .notify import weekly
    weekly(_stdin_json() or {})


@notify_app.command("nightly")
def notify_nightly() -> None:
    """Mail the nightly list sync's failures, if any (its JSON on stdin)."""
    _setup()
    from .notify import nightly
    nightly(_stdin_json() or [])


@notify_app.command("job-failed")
def notify_job_failed(job: str = typer.Option(..., help="backup, list_sync, monthly or weekly")
                      ) -> None:
    """Mail that a scheduled job failed; the tail of its log comes on stdin."""
    import sys
    _setup()
    from .notify import job_failed
    job_failed(job, sys.stdin.read())


@app.command("worker")
def worker_cmd(threads: int = typer.Option(None, help="tasks at once (default WORKER_THREADS)")
               ) -> None:
    """Run background tasks (rebuilds, list syncs, onboarding) until stopped."""
    _setup()
    from .tasks import worker
    worker(threads or settings().worker_threads)


@app.command("build-all")
def build_all_cmd(user: str = typer.Option(None), limit: int = typer.Option(50)) -> None:
    """Rebuild every surface and cache them."""
    _setup()
    typer.echo(json.dumps(build_all(_uid(user), limit), indent=2))


@app.command("feedback")
def feedback_cmd(mal_id: int, action: str, user: str = typer.Option(None)) -> None:
    """Record feedback, e.g. `malrec feedback 1535 not_interested`."""
    _setup(verbose=False)
    typer.echo(json.dumps(fb.record(_uid(user), mal_id, action), indent=2))


@app.command()
def status() -> None:
    """Show what is in the database."""
    _setup(verbose=False)
    typer.echo(f"anime           {scalar('SELECT count(*) FROM anime'):>8}")
    typer.echo(f"  graph fetched {scalar('SELECT count(*) FROM anime WHERE graph_fetched_at IS NOT NULL'):>8}")
    typer.echo(f"  anilist done  {scalar('SELECT count(*) FROM anime WHERE al_fetched_at IS NOT NULL'):>8}")
    typer.echo(f"  with tag_vec  {scalar('SELECT count(*) FROM anime WHERE tag_vec IS NOT NULL'):>8}")
    typer.echo(f"rec edges       {scalar('SELECT count(*) FROM rec_edge'):>8}")
    typer.echo(f"relations       {scalar('SELECT count(*) FROM relation'):>8}")
    typer.echo(f"franchises      {scalar('SELECT count(DISTINCT franchise_id) FROM franchise'):>8}")
    for u in query("SELECT u.mal_username, count(le.mal_id) n,"
                   " count(*) FILTER (WHERE le.score>0) s FROM app_user u"
                   " LEFT JOIN list_entry le ON le.user_id=u.id GROUP BY u.mal_username"):
        typer.echo(f"user {u['mal_username']}: {u['n']} entries, {u['s']} scored")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the HTTP API."""
    import uvicorn
    uvicorn.run("malrec.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
