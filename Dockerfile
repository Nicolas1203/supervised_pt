# Pull base pytorch image
FROM pytorch/pytorch

# Install repository specific dependencies
RUN pip install scipy
RUN pip install pandas
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install kneed
RUN apt update && apt upgrade -y
RUN apt install -y libwebp-dev
# Not sure these steps are necessary for libwebp to be installed
RUN conda install -c conda-forge libwebp \
    && conda install -c conda-forge/label/gcc7 libwebp \
    && conda install -c conda-forge/label/broken libwebp \
    && conda install -c conda-forge/label/cf201901 libwebp \
    && conda install -c conda-forge/label/cf202003 libwebp
# Need to reinstall pillow
RUN pip install --upgrade --force-reinstall pillow


# Define user to use (No root)
ARG HOST_USER_ID=37781
ARG HOST_USER_NAME=micheln
ARG HOST_GROUP_ID=513
ARG USER_DIR=/home/${HOST_USER_NAME}

RUN groupadd -g ${HOST_GROUP_ID} domain
RUN useradd -l -u ${HOST_USER_ID} -g domain ${HOST_USER_NAME}

#  Define working directory
WORKDIR ${USER_DIR}

# Fix permissions
RUN chown -R ${HOST_USER_NAME} ${USER_DIR}
RUN chgrp -R ${HOST_GROUP_ID} ${USER_DIR}

# Define mountable volumes
# Note VOLUME must be after the chown and chgrp for them to take effect
VOLUME ${USER_DIR}
