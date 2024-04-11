
ARG PYVERSION
FROM python:$PYVERSION
ARG TARGETPLATFORM

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install -y --no-install-recommends gdebi-core
RUN apt-get install -y jq
RUN apt-get install -y sudo

# Install gh
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
	&& chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh -y

RUN pip install --upgrade pip

# Install pydata dependencies
RUN pip install pandas matplotlib numpy scikit-learn pytest pytest-snapshot seaborn nbformat jupyter plotly black pytest-black \
    coverage pytest-cov umap-learn statsmodels tabulate rnanorm pydeseq2 quartodoc

# Install poetry 
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local/ python3 -

# Install quarto
RUN target=$(echo "$TARGETPLATFORM" | sed "s/\//-/") \
    && quarto_file=$(echo "https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-$target.deb") \
    && curl -LO "$quarto_file" && \
    gdebi --non-interactive quarto-1.3.450-$target.deb

# Install pydata
RUN git clone https://github.com/jsleight1/pydata.git \
	&& cd pydata \
	&& git checkout -b development origin/development \
	&& poetry update \
	&& poetry install \
	&& poetry build \
	&& pip install dist/pydata-$(poetry version -s)-py3-none-any.whl
