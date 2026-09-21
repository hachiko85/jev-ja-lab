from openjev_ja.common.revisions import local_git_revision


def test_local_git_revision_reads_loose_ref(tmp_path) -> None:
    git_dir = tmp_path / ".git"
    ref = git_dir / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    ref.write_text("abc123\n", encoding="utf-8")
    assert local_git_revision(str(tmp_path)) == "abc123"
