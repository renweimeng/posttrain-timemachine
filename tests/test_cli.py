from pathlib import Path

from typer.testing import CliRunner

from ptm.cli import app
from ptm.contracts import Origin, RewardContract, save_contract
from ptm.storage import Store


def test_documented_cli_workflow(tmp_path):
    runner = CliRunner()
    root = tmp_path/'data'
    args = ['--root', str(root)]
    assert runner.invoke(app, ['demo', '--no-serve', *args]).exit_code == 0
    assert runner.invoke(app, ['doctor', 'demo-format-truth', *args]).exit_code == 0
    spec = Path(__file__).resolve().parents[1]/'examples/format-over-truth/patch.yaml'
    analyzed = runner.invoke(app, ['analyze','reward-diff','--run','demo-format-truth','--spec',str(spec),*args])
    assert analyzed.exit_code == 0, analyzed.output
    import json
    aid = json.loads(analyzed.output)['analysis_id']
    report = tmp_path/'report.zip'
    assert runner.invoke(app, ['report','build','--analysis',aid,'--output',str(report),*args]).exit_code == 0
    assert runner.invoke(app, ['verify',str(report),'--recompute']).exit_code == 0
    store = Store(root)
    cs = store.load('demo-format-truth').completions[:2]
    save_contract(store, RewardContract(id='cli-order',fixtures=cs,
        assertions=[{'type':'reward_order','higher':cs[0].completion_id,'lower':cs[1].completion_id}],
        origin=Origin(run_id='demo-format-truth',reviewed=True)))
    contracts = root/'contracts'
    result = runner.invoke(app,['contract','test',str(contracts),'--reward-spec',str(spec),
        '--json-output',str(tmp_path/'results.json'),'--junit-output',str(tmp_path/'junit.xml')])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app,['contract','export-pytest',str(contracts),'--reward-spec',str(spec),
                               '-o',str(tmp_path/'test_reward.py')])
    assert result.exit_code == 0 and (tmp_path/'test_reward.py').exists()
