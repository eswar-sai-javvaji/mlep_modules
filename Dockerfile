FROM continuumio/miniconda3:latest
LABEL AUTHOR=sai_javvaji
RUN mkdir /mlep
ADD new_test_mlep /mlep
ENTRYPOINT ["/mlep/start_run.sh"]
