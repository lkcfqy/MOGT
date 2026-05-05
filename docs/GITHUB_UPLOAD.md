# GitHub Upload Notes

The repo is ready to push, but the current VM must have GitHub credentials.

Current remote:

```bash
git remote -v
```

Expected:

```text
origin  https://github.com/lkcfqy/MOGT.git (fetch)
origin  https://github.com/lkcfqy/MOGT.git (push)
```

## Recommended Upload

Install and authenticate GitHub CLI on a machine you trust:

```bash
gh auth login
git push origin main
```

If using a token over HTTPS, do not write the token into files. Use GitHub's
interactive credential prompt or a credential manager.

## SSH Alternative

Add an SSH key to GitHub, then switch the remote:

```bash
git remote set-url origin git@github.com:lkcfqy/MOGT.git
git push origin main
```

## Offline Backup

This VM also contains two local handoff artifacts:

```text
handoff_archives/mogt_paper_freeze_20260505.tar.gz
handoff_archives/mogt_git_freeze_20260505.bundle
```

The tarball is a source/evidence snapshot. The bundle is a portable Git archive:

```bash
git clone handoff_archives/mogt_git_freeze_20260505.bundle MOGT
cd MOGT
git remote set-url origin https://github.com/lkcfqy/MOGT.git
```

Large runtime directories are intentionally excluded from Git:

- `baseline_checkpoints/`
- `mogt_checkpoints/`
- `dataset_cache/`
- `profile_runs*/`
- logs, pid files, and model weights
