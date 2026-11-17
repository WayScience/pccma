# Deploy the pilot dashboard on merge

## What happens automatically

`.github/workflows/dashboard-pages.yml` runs when dashboard or workflow files
change in a pull request to `main`, when those changes land on `main`, and on
manual dispatch.

1. Run the exporter, input-packaging and browser fixture tests plus JavaScript
   tests with the locked dashboard environment.
2. Download the public pilot-input archive pinned by URL, exact byte count and
   SHA-256 in `pilot-source.json`. Reject missing, changed, unsafe or incomplete
   inputs; do not fall back to synthetic images or a coordinate-only report.
3. Build a fresh report from the checked-out Python, HTML, JavaScript and CSS.
   The input archive contains public report-input tables and original JPEGs,
   **not a prebuilt page**.
   Changes to channel controls therefore reach the deployed site on the next
   merge without republishing the image data.
4. Check every bundled JPEG against the original warehouse hashes and test the
   real pilot in Chromium, including hover/pin channel mixing, root and project
   URLs, mobile/file viewing, and the DuckDB catalog download.
5. On `WayScience/pccma`'s `main` only, upload the complete generated directory
   and deploy it to the `github-pages` environment. Pull requests build and test
   the same pilot but cannot publish it. A failed test/build blocks deployment.

The workflow uses no SSH access to Willow/bandicoot, local crop cache, PAT or
custom repository secrets. Its default permission is `contents: read`; only the
separate deployment job receives `pages: write` and `id-token: write`. Action
revisions are pinned to verified commit SHAs. No generated image tree, database
or archive should be committed to Git.

## One-time maintainer setup before merging

### 1. Publish the immutable input asset

Publishing the input archive makes the selected cell images public. Obtain the
normal dataset/publication approval before this step.

The prepared upload file (415,035,598 bytes) is:

```text
7.dashboard/dist/pilot-inputs.tar.gz
```

Public Parquet inputs retain all original rows and exact report-required values,
but omit unused source-directory columns and table attributes that could disclose
private HPC paths. `pilot-provenance.json` records the original and public input
hashes and omitted columns. Original JPEGs and the compressed JPEG hash list are
unchanged. The original private input files are not modified.

This is a **different artifact** from the earlier `BR00150695-pages-pilot-v4.zip`.
The ZIP is a rendered report for manual viewing; CI needs the public input
archive so it can build the latest application code.

Upload `pilot-inputs.tar.gz` to a **published, non-draft GitHub Release** with tag
`dashboard-pilot-BR00150695-v1` in `WayScience/pccma`. Keep the filename unchanged.
The release's tag is a data-snapshot identifier, not an application version.
From the repository root, an authenticated maintainer can do this with:

```bash
gh release create dashboard-pilot-BR00150695-v1 \
  7.dashboard/dist/pilot-inputs.tar.gz \
  --repo WayScience/pccma \
  --target main \
  --title "Dashboard pilot inputs: BR00150695" \
  --notes "Immutable public report tables, original warehouse hashes and selected JPEGs for the 24-hour BR00150695 pilot."
```

Alternatively, create the release and upload the asset in GitHub's Releases UI.
Do not upload the rendered-report ZIP under the input archive's name. Do not
replace this asset with different bytes; a checksum mismatch intentionally
fails CI. For new data, publish a new snapshot tag and update the reviewed URL,
SHA-256, size and scope in `pilot-source.json` together.

Check the published asset before merging by downloading, validating and staging
it into a new local directory:

```bash
uv run --project 7.dashboard --locked python 7.dashboard/pages.py stage \
  --config 7.dashboard/pilot-source.json \
  --output 7.dashboard/previews/release-input-check
```

The upload is a one-time operation. Subsequent UI/code merges reuse this pinned
public data asset; there is no dependency on expiring Actions artifacts.

### 2. Enable GitHub Pages using Actions

In `WayScience/pccma`, open:

```text
Settings → Pages → Build and deployment → Source → GitHub Actions
```

Do not choose deployment from a branch or select the repository root as a
Jekyll source. This application is generated from a separate build directory.
The workflow checks that Pages is configured; it does not silently create or
reconfigure the public site.

Actions were enabled for the repository when this workflow was prepared. If an
organization policy changes, allow the pinned `actions/*` and
`astral-sh/setup-uv` actions. Do not add a personal-access-token secret or broaden
all jobs' token permissions.

### 3. Check environment deployment rules

Under `Settings → Environments`, create or inspect `github-pages`.
For branch restriction, use **Selected branches and tags → branch `main`**.
Do not select "protected branches only" unless `main` is actually protected.
Leave required reviewers off for fully automatic deployment; if reviewers are
required, a merge builds the site but publication waits for approval.

### 4. Review, push and merge the code changes

Include the workflow, `7.dashboard/pages.py`, `pilot-source.json`,
`pilot-provenance.json`, packaging tests, source-hash test update and
documentation in the change. Keep `dist/` and
`previews/` ignored. Upload the public data asset before relying on the PR build
check: the real-data job deliberately fails if the asset is absent.

Wait for both `Dashboard tests` and `Build verified pilot` to pass on the PR.
These can be made required checks in branch protection. A merge that changes
the dashboard then runs the same checks and deploys automatically. Unrelated
changes outside the workflow's path filters do not republish the site.

The expected default project-site URL after a successful deployment is:

```text
https://wayscience.github.io/pccma/
```

The workflow's deployment environment reports the authoritative URL, including
any custom domain configured later. After the first merge, inspect that exact
run's successful deployment and open the hosted page to verify real image
loading and hover/pin comparison. Local checks cannot prove a hosted deployment.

If the first run failed because setup was incomplete, finish the setup and
rerun the failed workflow. Once the workflow is on `main`, a manual rebuild is:

```bash
gh workflow run dashboard-pages.yml --repo WayScience/pccma --ref main
```

## Reproduce the build without publishing

Use an empty output directory for each stage/build; neither command overwrites
an existing successful result.

To use the prepared local archive without downloading it (the same checksum
and safety checks still apply):

```bash
uv run --project 7.dashboard --locked python 7.dashboard/pages.py stage \
  --config 7.dashboard/pilot-source.json \
  --archive 7.dashboard/dist/pilot-inputs.tar.gz \
  --output 7.dashboard/previews/pages-inputs

uv run --project 7.dashboard --locked python 7.dashboard/pages.py build \
  --config 7.dashboard/pilot-source.json \
  --inputs 7.dashboard/previews/pages-inputs \
  --output 7.dashboard/dist/pages-rehearsal

PCCMA_DASHBOARD_BUNDLE="$PWD/7.dashboard/dist/pages-rehearsal" \
PCCMA_DASHBOARD_SOURCE_HASHES="$PWD/7.dashboard/previews/pages-inputs/warehouse/jpg-sha256.txt.gz" \
uv run --project 7.dashboard --locked python -m pytest \
  7.dashboard/tests/test_real_bundle.py -q
```

For a new snapshot, regenerate the archive from read-only original inputs using
`pages.py package`, then review its printed descriptor before updating the
configuration. Do not pin a checksum guessed from a filename or an archive that
was not actually produced and verified.
## Verification performed before handoff

A clean Linux x86_64 container (Python 3.12, Node 24, locked uv environment)
completed the source/fixture suite: **137 passed**, plus **19 JavaScript tests**.
Six fixture-stage tests skipped as designed: five real-data tests awaited the
build, and one optional test required an existing private cache.

The same container then staged the actual pinned public archive, built the
6,500-cell/39,000-JPEG pilot without local cache access, and passed all **five
real-bundle tests** using the staged original hash list. A GNU tar Pages artifact
was created and read back: **452,751,360 bytes / 39,004 files**, with image/catalog
hashes, HTML hash, `.nojekyll` and no-link checks passing. `actionlint` and
`zizmor` also passed for the workflow.

The rehearsal used `--archive` for local acquisition; HTTPS acquisition has
fixture coverage, but the public release URL cannot be exercised until a
maintainer uploads the asset. No release, repository setting, commit, push or
hosted deployment was performed as part of this preparation.
