# Timeseries tutorial

### python execution and debug
Python interpreter: Docker compose -> GPU, Dockerfile -> nonGPU

### jupyter
* Make container for jupyter or playground  
```
$ docker-compose run -p 8003:8003 --name=timeseries_forecasting_master_run timeseries_forecasting_master bash
```

* Settings on docker
```
# jupyter notebook --generate-config
# jupyter notebook password
```

* launch
```
# jupyter lab --ip=0.0.0.0 --port=8003 --allow-root
```

### tensorboard
```
# tensorboard --logdir (path/to/the/logs) --port 6006 --host 0.0.0.0