const { spawn } = require('child_process');
const path = require('path');

const proc = spawn('python', [path.join(__dirname, 'feishu_poll.py')], {
  cwd: path.join(__dirname, '..'),
  stdio: 'inherit',
  windowsHide: true,
  env: { ...process.env, PYTHONUNBUFFERED: '1' }
});
proc.on('close', (code) => process.exit(code));
