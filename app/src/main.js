const { app, Tray, Menu, BrowserWindow, Notification, ipcMain, nativeImage } = require('electron');
const path = require('path');
const { Worker } = require('./worker');

require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

// Singleton — only one instance allowed
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
}

let tray = null;
let mainWindow = null;
let worker = null;

// If a second instance tries to launch, show the existing window
app.on('second-instance', () => {
  if (mainWindow) {
    mainWindow.show();
    mainWindow.focus();
  }
});

function createTrayIcon() {
  const iconPath = path.join(__dirname, '..', 'assets', 'tray-icon.png');
  let icon;
  try {
    icon = nativeImage.createFromPath(iconPath);
    if (icon.isEmpty()) throw new Error('empty');
  } catch {
    const size = 16;
    const buf = Buffer.alloc(size * size * 4);
    for (let i = 0; i < size * size; i++) {
      const x = i % size;
      const y = Math.floor(i / size);
      const cx = size / 2, cy = size / 2, r = size / 2 - 1;
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (dist <= r) {
        buf[i * 4] = 0;
        buf[i * 4 + 1] = 200;
        buf[i * 4 + 2] = 120;
        buf[i * 4 + 3] = 255;
      }
    }
    icon = nativeImage.createFromBuffer(buf, { width: size, height: size });
  }
  return icon;
}

function createWindow() {
  if (mainWindow) {
    mainWindow.show();
    mainWindow.focus();
    return;
  }

  mainWindow = new BrowserWindow({
    width: 500,
    height: 600,
    resizable: true,
    skipTaskbar: true,
    frame: true,
    title: 'GenShape3D Worker',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'ui', 'index.html'));

  mainWindow.on('close', (e) => {
    e.preventDefault();
    mainWindow.hide();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function showNotification(title, body) {
  if (Notification.isSupported()) {
    new Notification({ title, body }).show();
  }
}

function updateTrayTooltip() {
  if (!tray || !worker) return;
  const state = worker.getState();

  if (state.currentJob) {
    const prog = state.currentJob.progress;
    if (prog && prog.pct > 0) {
      const detail = prog.detail || prog.phase;
      tray.setToolTip(`GenShape3D - ${prog.pct}% - ${detail}`);
    } else {
      tray.setToolTip(`GenShape3D - Processing job...`);
    }
  } else if (state.pendingJobs.length > 0) {
    tray.setToolTip(`GenShape3D - ${state.pendingJobs.length} pending`);
  } else {
    tray.setToolTip('GenShape3D Worker - Idle');
  }
}

app.on('ready', () => {
  if (process.platform === 'win32') {
    app.setAppUserModelId('GenShape3D Worker');
  }

  const icon = createTrayIcon();
  tray = new Tray(icon);
  tray.setToolTip('GenShape3D Worker - Starting...');

  function buildTrayMenu() {
    const maxJobs = worker ? worker.maxConcurrent : 1;
    return Menu.buildFromTemplate([
      { label: 'Show Jobs', click: () => createWindow() },
      { type: 'separator' },
      { label: 'Max Concurrent Jobs', enabled: false },
      ...[1, 2, 3, 4].map(n => ({
        label: `  ${n} job${n > 1 ? 's' : ''}`,
        type: 'radio',
        checked: maxJobs === n,
        click: () => {
          if (worker) worker.maxConcurrent = n;
          tray.setContextMenu(buildTrayMenu());
        },
      })),
      { type: 'separator' },
      { label: 'Quit', click: () => {
        if (mainWindow) {
          mainWindow.removeAllListeners('close');
          mainWindow.close();
        }
        if (worker) worker.stop();
        app.quit();
      }},
    ]);
  }

  tray.setContextMenu(buildTrayMenu());
  tray.on('click', () => createWindow());

  // Start the worker
  worker = new Worker({
    databaseUrl: process.env.DATABASE_URL,
    r2Endpoint: process.env.R2_ENDPOINT || 'https://edad30fa0fe66f50971087c6b0df0f28.r2.cloudflarestorage.com',
    r2AccessKey: process.env.R2_ACCESS_KEY_ID || '',
    r2SecretKey: process.env.R2_SECRET_ACCESS_KEY || '',
    r2Bucket: process.env.R2_BUCKET || 'genshape3d',
    r2PublicUrl: process.env.R2_PUBLIC_URL || '',
    pollInterval: parseInt(process.env.POLL_INTERVAL || '10000', 10),
    // Models this worker can run. The 1080 has only Hunyuan3D; the 3090
    // declares 'hunyuan3d,triposr,sf3d,hi3dgen'. worker.js's poll filters
    // pending by this list — without it, .env's WORKER_MODELS was ignored.
    models: (process.env.WORKER_MODELS || 'hunyuan3d').split(',').map(s => s.trim()).filter(Boolean),
  });

  worker.on('jobReceived', (job) => {
    showNotification('Job Received', `Job ${job.id.slice(0, 8)}... is queued`);
    sendToRenderer('job-update', worker.getState());
    updateTrayTooltip();
  });

  worker.on('jobProcessing', (job) => {
    const startTime = new Date(job.startedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    showNotification('Job Started', `Job ${job.id.slice(0, 8)}... started at ${startTime}`);
    sendToRenderer('job-update', worker.getState());
    updateTrayTooltip();
  });

  worker.on('progressUpdate', (progress) => {
    updateTrayTooltip();
    sendToRenderer('job-update', worker.getState());
  });

  worker.on('jobComplete', (job) => {
    const started = new Date(job.startedAt).getTime();
    const ended = new Date(job.completedAt).getTime();
    const duration = Math.round((ended - started) / 1000);
    showNotification('Complete', `Job ${job.id.slice(0, 8)}... done in ${duration}s`);
    sendToRenderer('job-update', worker.getState());
    updateTrayTooltip();
  });

  worker.on('jobFailed', (job, error) => {
    showNotification('Failed', `Job ${job.id.slice(0, 8)}... failed: ${error.slice(0, 80)}`);
    sendToRenderer('job-update', worker.getState());
    updateTrayTooltip();
  });

  worker.on('stateChanged', () => {
    sendToRenderer('job-update', worker.getState());
    updateTrayTooltip();
  });

  worker.start();

  // IPC handlers
  ipcMain.handle('get-state', () => worker.getState());
  ipcMain.handle('cancel-job', (_e, jobId) => worker.cancelJob(jobId));
});

function sendToRenderer(channel, data) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, data);
  }
}

app.on('window-all-closed', (e) => {
  // Don't quit - keep running in tray
});
