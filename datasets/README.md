# Datasets

This directory stores datasets required for this project.

⚠️ **Important**
- Dataset files are **NOT tracked by Git**
- Only this `README.md` is version-controlled
- All datasets are hosted on **Microsoft OneDrive**
- Data access and download are handled via **rclone**

---

## 1. Prerequisites

### Install rclone

#### Windows
Download rclone from:
https://rclone.org/download/

Unzip `rclone.exe` and make sure it is added to your system `PATH`.

Verify installation:
```bash
rclone version
```

#### Linux / macOS
```bash
curl https://rclone.org/install.sh | sudo bash
```

---

## 2. Configure OneDrive Remote

Run the interactive configuration:
```bash
rclone config
```

Recommended steps:
1. Choose **New remote**
2. Name the remote: `onedrive`
3. Select storage type: **OneDrive**
4. Complete browser-based authentication
5. Choose **OneDrive Personal** or **Business** as appropriate

Test the configuration:
```bash
rclone lsd onedrive:
```

---

## 3. List Available Datasets on OneDrive

List top-level directories:
```bash
rclone lsd onedrive:
```

List files inside a dataset directory:
```bash
rclone ls onedrive:/path/to/dataset
```

Example:
```bash
rclone lsd onedrive:/NeuromorphicCamera
rclone ls  onedrive:/NeuromorphicCamera/event_data
```

---

## 4. Download Datasets

### Download an Entire Dataset Folder
```bash
rclone copy onedrive:/NeuromorphicCamera/event_data ./event_data
```

### Download with Progress Bar
```bash
rclone copy onedrive:/NeuromorphicCamera/event_data ./event_data -P
```

### Resume Interrupted Downloads
```bash
rclone copy onedrive:/NeuromorphicCamera/event_data ./event_data \
  --progress \
  --transfers 4
```

---

## 5. Verify Downloaded Data

Check remote dataset size:
```bash
rclone size onedrive:/NeuromorphicCamera/event_data
```

Check local directory size:
```bash
du -sh ./event_data
```

---

## 6. Usage Notes

- ❌ Do NOT commit dataset files to Git
- ✅ Keep large datasets on OneDrive only
- ✅ Use `rclone copy` instead of `sync` to avoid accidental deletion
- 📁 Dataset paths and names may change — always verify using `rclone lsd`

---

## 7. Optional: Mount OneDrive (Advanced)

Mount OneDrive as a local drive (read-only recommended):

### Windows
```bash
rclone mount onedrive:/NeuromorphicCamera X: --read-only
```

### Linux / macOS
```bash
rclone mount onedrive:/NeuromorphicCamera ~/onedrive --read-only
```

Stop mounting:
```bash
Ctrl + C
```

---

## Contact

If you encounter access issues or missing datasets, please contact the project maintainer.
