const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  getState: () => ipcRenderer.invoke('get-state'),
  cancelJob: (jobId) => ipcRenderer.invoke('cancel-job', jobId),
  onJobUpdate: (cb) => {
    ipcRenderer.on('job-update', (_e, data) => cb(data));
  },
});
