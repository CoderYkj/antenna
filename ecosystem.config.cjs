module.exports = {
  apps: [
    {
      name: 'antenna-bot',
      cwd: 'e:/antenna/server',
      script: 'start.cjs',
      interpreter: 'C:/Program Files/nodejs/node.exe',
      restart_delay: 5000,
      max_restarts: 10,
      env: {
        PYTHONUNBUFFERED: '1'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: 'e:/antenna/logs/pm2-error.log',
      out_file: 'e:/antenna/logs/pm2-out.log'
    },
    {
      name: 'pm2-monitor',
      cwd: 'e:/antenna',
      script: 'server/pm2_monitor.py',
      interpreter: 'C:/Python314/python.exe',
      restart_delay: 5000,
      max_restarts: 10,
      env: {
        PYTHONUNBUFFERED: '1'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: 'e:/antenna/logs/pm2-monitor-error.log',
      out_file: 'e:/antenna/logs/pm2-monitor-out.log'
    }
  ]
}
