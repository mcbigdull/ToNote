"""Sequential local batch runner with durable logs and a heartbeat status."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VOCAL = Path(__file__).resolve().parent
STAGES = [('prepare', '解码原曲 / HTDemucs 分离'), ('bs', 'BS-RoFormer 分离'),
          ('analyze', '生成三路音高数据'), ('render', '绘制热图 / 生成同步 HTML')]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # One batch at a time; OS releases the byte lock even if the runner crashes.
    import msvcrt
    lock = (VOCAL / 'output' / '.comparison_batch.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    songs = sorted((VOCAL / 'input').glob('*.mp3'))
    if not songs:
        raise RuntimeError('No MP3 files in vocal/input')
    state = dict(status='running', started=now(), pid=os.getpid(),
                 log=str(folder / 'progress.log'), completed=0, failed=0,
                 songs=[dict(input=str(song), status='pending',
                             result=str(VOCAL / 'output' / (song.stem + '_separation_compare') / 'index.html'))
                        for song in songs])

    def save():
        state['updated'] = now()
        for name, text in (
            ('status.json', json.dumps(state, ensure_ascii=False, indent=2)),
            ('status.txt', '\n'.join([
                f"状态：{state['status']}", f"最近更新：{state['updated']}",
                f"完成：{state['completed']}/{len(songs)}；失败：{state['failed']}",
                f"当前歌曲：{state.get('current_song', '—')}",
                f"当前阶段：{state.get('stage', '—')}",
                f"日志：{state['log']}",
                *[f"{x['status']} | {Path(x['input']).name} | {x['result']}" for x in state['songs']]
            ]))):
            temporary = folder / (name + '.tmp')
            temporary.write_text(text, encoding='utf-8')
            temporary.replace(folder / name)

    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    child = None
    save()
    with (folder / 'progress.log').open('a', encoding='utf-8', buffering=1) as log:
        try:
            for song, item in zip(songs, state['songs']):
                state['current_song'] = song.name
                item['status'] = 'running'
                for stage, label in STAGES:
                    state['stage'] = label
                    log.write(f'\n[{now()}] {song.name} | {label}\n')
                    log.flush()
                    child = subprocess.Popen([
                        str(VOCAL.parent / '.venv' / 'Scripts' / 'python.exe'), '-u',
                        str(VOCAL / 'compare_separation.py'), str(song),
                        '--resume', '--stage', stage
                    ], cwd=VOCAL.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    save()
                    while child.poll() is None:
                        time.sleep(10)
                        save()
                    code = child.returncode
                    state.pop('child_pid', None)
                    if code:
                        item.update(status='failed', failed_stage=stage, exit_code=code)
                        state['failed'] += 1
                        log.write(f'[{now()}] FAILED: exit code {code}; continuing next song\n')
                        break
                else:
                    item['status'] = 'completed'
                    state['completed'] += 1
                    log.write(f'[{now()}] COMPLETED: {song.name}\n')
                save()
            state.update(status='completed' if not state['failed'] else 'completed_with_errors',
                         stage='批处理结束', finished=now())
            save()
            log.write(f'[{now()}] BATCH FINISHED: {state["status"]}\n')
        except BaseException as exc:
            if child is not None and child.poll() is None:
                state['child_pid'] = child.pid
            state.update(status='failed', error=repr(exc), finished=now())
            log.write(f'[{now()}] RUNNER ERROR: {exc!r}\n')
            save()
            raise


if __name__ == '__main__':
    main()
