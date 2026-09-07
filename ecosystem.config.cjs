const path = require('path');

const rootDir = __dirname;
const logsDir = path.join(rootDir, 'logs');
const isWindows = process.platform === 'win32';
const pythonBin = process.env.ANTENNA_PYTHON || (isWindows ? 'python' : 'python3');

module.exports = {
  apps: [
    {
      name: 'antenna-bot',
      cwd: rootDir,
      script: 'server/feishu_poll.py',
      interpreter: pythonBin,
      restart_delay: 5000,
      max_restarts: 10,
      stop_exit_codes: [2],
      env: {
        PYTHONUNBUFFERED: '1',
        ANTENNA_PYTHON: pythonBin,
        TQDM_DISABLE: '1'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: path.join(logsDir, 'pm2-error.log'),
      out_file: path.join(logsDir, 'pm2-out.log')
    },
    {
      name: 'pm2-monitor',
      cwd: rootDir,
      script: 'server/pm2_monitor.py',
      interpreter: pythonBin,
      restart_delay: 5000,
      max_restarts: 10,
      env: {
        PYTHONUNBUFFERED: '1',
        TQDM_DISABLE: '1'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: path.join(logsDir, 'pm2-monitor-error.log'),
      out_file: path.join(logsDir, 'pm2-monitor-out.log')
    }
  ]
};
