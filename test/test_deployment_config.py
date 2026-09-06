import pytest
from shuiyuan_auto_reply.bootstrap import deployment

@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(deployment, '_current', None)
    monkeypatch.setattr(deployment, 'load_dotenv', lambda: None)
    monkeypatch.setattr(deployment.os, 'environ', {})

def test_precedence_and_relative_paths(tmp_path):
    deployment.os.environ['EMBEDDING_DIMS'] = '256'
    config = tmp_path / 'config.toml'
    config.write_text('[common.embedding]\ndims=512\n[profiles.local.embedding]\ndims=768\n[common.forum]\ncookie_file="secret.json"\n')
    result = deployment.load_deployment(str(config), overrides={'embedding': {'dims': 1024}})
    assert result.section('embedding')['dims'] == 1024
    assert result.section('forum')['cookie_file'] == str(tmp_path / 'secret.json')

def test_secret_redaction(tmp_path):
    (tmp_path / 'key').write_text('sensitive')
    config = tmp_path / 'config.toml'
    config.write_text('[common.embedding]\napi_key={file="key"}\n')
    result = deployment.load_deployment(str(config))
    assert result.section('embedding')['api_key'] == 'sensitive'
    assert 'sensitive' not in str(result.redacted())

def test_remote_requires_file():
    with pytest.raises(ValueError):
        deployment.load_deployment(profile='remote')
