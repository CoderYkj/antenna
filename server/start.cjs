const { spawn } = require('child_process');
const path = require('path');

const pythonBin = process.env.ANTENNA_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

const proc = spawn(pythonBin, [path.join(__dirname, 'feishu_poll.py')], {
  cwd: path.join(__dirname, '..'),
  stdio: 'inherit',
  windowsHide: true,
  env: { ...process.env, PYTHONUNBUFFERED: '1' }
});
proc.on('close', (code) => process.exit(code));
