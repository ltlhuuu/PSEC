## Process dataset
This code can filter the dataset with your preference. You can fitler the trajectories with high rewars and low costs. You can run ```process_dataset.py```.

### env_name.py
This code includes the env tasks list.

### utils.py
This file includes the necessary function used to finish the process dataset.

```
python /home/liutl/work/lora_meta/LoRaMeta/env/process_dataset.py  --env_id 2 --save --cmax 2 --cmin 0 --rmax 100 --rmin 93
```