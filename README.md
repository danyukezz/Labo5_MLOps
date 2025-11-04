# Labo5_MLOps / 01_DockerTest

This repo contains a minimal Dockerfile and a static `index.html` served by nginx.

Files added:

- `Dockerfile` — uses `nginx:alpine` and copies the repo contents into nginx html folder.
- `index.html` — basic "Hello world!" page.

How to build and test locally:

1. Build the image:

   ```
   docker build -t hello-world-nginx:v1 .
   ```

2. Run the container mapping host port 8080 to container port 80:

   ```
   docker run -d -p 8080:80 --name hello-nginx hello-world-nginx:v1
   ```

3. Open http://localhost:8080 in your browser.

Notes about GitHub steps:

- Create a new repo on GitHub under your account (e.g. `01_DockerTest-Repo`).
- Add the repo as a remote and push the `01_DockerTest` branch:

  ```
  git remote add origin https://github.com/<your-username>/<repo-name>.git
  git push -u origin 01_DockerTest
  ```

# CI/CD Pipeline Test
