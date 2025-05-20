import os
import json
tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
task_config = tf_config.get('task', {})
task_type = task_config.get('type')
task_index = task_config.get('index')
  
job_name = task_type
task_index = task_index

if not tf_config:
    job_name = 'worker'
    task_index = 0
print(job_name+'_'+str(task_index))

