FROM hub.c.163.com/library/python:3.6
FROM hub-mirror.c.163.com/ufoym/deepo

#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
#ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs


ADD ./data/ test/data/
ADD ./Model/ test/Model/
ADD ./dataset.py test/
ADD ./inference.py test/
ADD ./inference.sh test/
ADD ./inference_cnn.py test/
ADD ./inference_crnn.py test/
ADD ./inference_model_fusion.py test/
ADD ./inference_res2net.py test/
ADD ./metrics.py test/
ADD ./torch_func.py test/
ADD ./utils.py test/
ADD ./requirements.txt test/
ADD ./run_res2net.py test/
ADD ./run_cnn.py test/
ADD ./run_crnn.py test/
ADD ./train_model_fusion.py test/
ADD ./run.sh test/

#RUN pip install -r test/requirements.txt

WORKDIR test

CMD echo start | tee start_ && sh run.sh && echo done |tee done_ && sleep 2h 
