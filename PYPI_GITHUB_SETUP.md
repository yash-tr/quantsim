# PyPI GitHub Actions Trusted Publisher Setup

This guide explains how to configure PyPI Trusted Publisher for automatic package publishing from GitHub Actions.

## 🔐 **What is PyPI Trusted Publisher?**

PyPI Trusted Publisher is a secure method that allows GitHub Actions to publish packages to PyPI without storing API tokens. It uses OpenID Connect (OIDC) for authentication.

**Benefits:**
- ✅ No API tokens to manage or rotate
- ✅ More secure than storing secrets
- ✅ Automatic token generation per workflow run
- ✅ Fine-grained permissions per repository

---

## 📋 **Step-by-Step Setup**

### **Step 1: GitHub Repository Setup**

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add GitHub Actions workflows for PyPI publishing"
   git push origin main
   ```

2. **Create a GitHub Environment (Recommended):**
   - Go to your repository settings
   - Navigate to **Environments**
   - Click **New environment**
   - Name it `pypi`
   - Add protection rules (optional but recommended):
     - ✅ Required reviewers (for additional security)
     - ✅ Restrict to protected branches

### **Step 2: PyPI Trusted Publisher Configuration**

1. **Go to PyPI:** https://pypi.org/manage/account/publishing/

2. **Add a new publisher** with these exact values:

   | Field | Value |
   |-------|-------|
   | **PyPI Project Name** | `quantsim` |
   | **Owner** | `yasht1004` (your GitHub username) |
   | **Repository name** | `quantsim` (your repo name) |
   | **Workflow name** | `publish-to-pypi.yml` |
   | **Environment name** | `pypi` (optional but recommended) |

3. **Click "Add"** to save the configuration

### **Step 3: First Release**

1. **Create a release tag:**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Create a GitHub Release:**
   - Go to your repository
   - Click **Releases** → **Create a new release**
   - Choose tag: `v0.1.0`
   - Title: `QuantSim v0.1.0 - Initial Release`
   - Description: Add release notes
   - ✅ Set as latest release
   - Click **Publish release**

3. **Monitor the workflow:**
   - Go to **Actions** tab in your repository
   - Watch the "Publish to PyPI" workflow run
   - Check for any errors in the logs

---

## 🔧 **Workflow Configuration Details**

### **Files Created:**

1. **`.github/workflows/publish-to-pypi.yml`**
   - Triggers on release publication
   - Builds and publishes to PyPI
   - Includes version verification and testing

2. **`.github/workflows/test.yml`**
   - Runs tests on multiple Python versions
   - Code quality checks
   - Security scans

### **Key Features:**

✅ **Automatic version verification** - Ensures package version matches release tag
✅ **Multi-step validation** - Build → Test → Verify → Publish
✅ **Cross-platform testing** - Ubuntu, Windows, macOS
✅ **Security scanning** - Dependency and code security checks
✅ **Manual trigger option** - Can run workflows manually if needed

---

## 🚀 **Publishing Process**

### **Automatic Publishing (Recommended):**

1. **Update version** in both files:
   - `quantsim/__init__.py`: `__version__ = "0.1.1"`
   - `pyproject.toml`: `version = "0.1.1"`

2. **Commit and push changes:**
   ```bash
   git add .
   git commit -m "Bump version to 0.1.1"
   git push origin main
   ```

3. **Create and push tag:**
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

4. **Create GitHub Release:**
   - The workflow will automatically trigger
   - Package will be built and published to PyPI

### **Manual Publishing (Fallback):**

If you need to publish manually:

```bash
# Build the package
python -m build

# Upload to PyPI (you'll need an API token)
twine upload dist/*
```

---

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **"Trusted publisher not found"**
   - ✅ Check PyPI configuration matches exactly
   - ✅ Ensure environment name matches (`pypi`)
   - ✅ Verify repository and owner names

2. **"Version already exists"**
   - ✅ Increment version number
   - ✅ Check both `__init__.py` and `pyproject.toml`

3. **"Workflow not authorized"**
   - ✅ Check repository permissions
   - ✅ Ensure environment is configured
   - ✅ Verify workflow file name matches PyPI config

4. **"Build failures"**
   - ✅ Check dependencies in `pyproject.toml`
   - ✅ Run tests locally first
   - ✅ Review workflow logs for details

### **Debug Commands:**

```bash
# Test build locally
python -m build

# Verify package
python -m twine check dist/*

# Test installation
pip install dist/*.whl

# Check imports
python -c "import quantsim; print(quantsim.__version__)"
```

---

## 📊 **Security Best Practices**

### **Implemented Security Measures:**

1. **OIDC Authentication** - No stored secrets
2. **Environment Protection** - Additional approval layer
3. **Version Verification** - Prevents accidental publishes
4. **Package Validation** - Ensures quality before publish
5. **Security Scanning** - Automated vulnerability checks

### **Additional Recommendations:**

- ✅ Enable branch protection on `main`
- ✅ Require PR reviews for changes
- ✅ Use environment protection rules
- ✅ Monitor PyPI downloads and usage
- ✅ Set up notification for releases

---

## 📈 **Monitoring and Maintenance**

### **After Setup:**

1. **Monitor first release** carefully
2. **Check PyPI package page** for proper display
3. **Test installation** from PyPI: `pip install quantsim`
4. **Set up notifications** for workflow failures
5. **Review security alerts** regularly

### **Regular Maintenance:**

- Update dependencies regularly
- Monitor security alerts
- Review and update workflows
- Check PyPI project page for issues
- Monitor download statistics

---

## ✅ **Quick Setup Checklist**

- [ ] GitHub repository created and code pushed
- [ ] Environment `pypi` created in GitHub
- [ ] PyPI Trusted Publisher configured
- [ ] Version numbers synchronized
- [ ] First tag created (`v0.1.0`)
- [ ] GitHub Release published
- [ ] Workflow execution successful
- [ ] Package available on PyPI
- [ ] Installation tested: `pip install quantsim`

---

## 🎯 **Next Steps After Setup**

1. **Promote your package** on relevant communities
2. **Set up documentation** hosting (Read the Docs)
3. **Add badges** to README (build status, PyPI version)
4. **Monitor issues** and user feedback
5. **Plan feature roadmap** for future releases

**Your package is now ready for automated PyPI publishing! 🚀** 