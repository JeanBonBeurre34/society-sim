FROM python:3.11-slim

WORKDIR /app

# only Python dependencies now
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir jupyterlab ipywidgets

COPY agent_brain.py society_sim.py ./

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
