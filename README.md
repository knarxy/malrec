# malrec

**Anime recommendations that learn your taste from your MyAnimeList ratings, and show their working.**

![Safe Bets in malrec](docs/screenshot.png)

Sign in with MyAnimeList and malrec ranks the rest of the catalogue against
what you actually rate highly, not just against what is popular.

- **Two layers, blended by how well it knows you.** A population model learned
  from thousands of public MAL lists carries thin profiles. A personal model,
  fitted to your own ratings, takes over as your list grows.
- **Every pick explains itself.** "Why this?" shows what your own ratings say,
  what viewers with your taste say, how the two are weighed, and why a title
  sits where it does in the list. Picks that rest on acclaim alone, with
  nothing on your list backing them, are held back.
- **Lists for different moods.** Safe Bets, Discover, Hidden Gems, Next Up
  (the next season of what you finished), This Season, Coming Soon and Side
  Stories.
- **Writes back to MAL.** Plan to watch and rate titles from the app. A
  quick rating round sharpens a new profile.
- **Watch together.** Pair with a friend for a list you would both enjoy.
  Only predicted scores are shared, never your lists.
- English and German.

![The "Why this?" panel](docs/why-this.png)

## Try it

**<https://mal.dschw.ninja>**: sign in with your MyAnimeList account.

New accounts have to be **approved** before recommendations are built, so
after your first sign-in you will see a waiting screen until that happens.

## Self-hosting

You need Docker with Compose and a MyAnimeList API client, which you can
register at <https://myanimelist.net/apiconfig>. Set its *App Redirect URL*
to `http://localhost:3000/api/auth/callback`, or to your own domain's
`/api/auth/callback`.

```bash
git clone https://github.com/knarxy/malrec.git && cd malrec
cp .env.example .env
```

In `.env`, set:

- `MAL_CLIENT_ID` and `MAL_CLIENT_SECRET`, from your MAL API client
- `ADMIN_USERS` and `MALREC_USER`, set to your MAL username (the admin approves new accounts)
- `TOKEN_KEY`, the key that encrypts stored MAL tokens. Generate one with
  `python3 -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"`
- `POSTGRES_PASSWORD`, set to something other than the default
- for a public domain behind HTTPS: `MAL_REDIRECT_URI`, `APP_BASE_URL` and
  `SESSION_COOKIE_SECURE=true`

Then start it and build the population model once. The fetch reads a sample
of public MAL lists; it takes a few hours and can be resumed:

```bash
docker compose up -d --build
docker compose --profile lab run --rm lab malrec sync full
docker compose --profile lab run --rm lab malrec population fit
```

Open <http://localhost:3000> and sign in. Your own account is approved
automatically because it is listed in `ADMIN_USERS`. You approve everyone
else from the admin panel in the account menu.

Everything binds to `127.0.0.1` by default. For public access, put a
TLS-terminating reverse proxy in front of port 3000. The optional nightly list
sync, backups and monthly refresh are in `scripts/malrec.cron`.

## How it works

- [docs/MODEL.md](docs/MODEL.md) covers the model, the ranking, the
  evaluation behind every setting, the configuration and the operations.
- [docs/RESEARCH.md](docs/RESEARCH.md) covers the literature behind the
  design choices.
- [experiments/](experiments) holds the experiment code and the logs of
  every measured change.

Built with FastAPI, scikit-learn, Postgres 17 with pgvector, React and Vite.
Data comes from the [MyAnimeList API](https://myanimelist.net/apiconfig/references/api/v2)
and [AniList](https://anilist.co).

## License

[MIT](LICENSE)
