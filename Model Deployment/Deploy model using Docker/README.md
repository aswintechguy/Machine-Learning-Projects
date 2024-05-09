1. To build the docker image, use the following command. Run this in the folder which has dockerfile.
```
docker build -t iris-flask-app .
```

2. To run the docker image, use the following command
```
docker run -d -p 5000:5000 iris-flask-app
```