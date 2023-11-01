# Detecção de Fraudes em Transações de E-commerce



## Introdução

O presente trabalho tem como objetivo realizar uma análise exploratória dos dados de transações financeiras em um e-commerce e construir um modelo de aprendizagem de máquina para detecção de fraudes financeiras.

* Todas as análises e processo de modelagem encontra-se no seguinte [notebook](https://github.com/francismelojr/fraud-modeling/blob/master/src/notebook/model.ipynb).

* O projeto foi desenvolvido utilizando Poetry para gerenciamento de dependências e Pyenv para controle de versão do Python. Também foi utilizado Docker para rodar a API de forma isolada. 

* Para replicação do projeto, é necessário instalar o [Poetry](https://python-poetry.org/docs/#installation) e [Pyenv](https://github.com/pyenv/pyenv/#installation). Caso queira utilizar o Docker, instale-o [aqui](https://docs.docker.com/desktop/).


## Instalação e configuração

### 1. Clone o repositório

```bash
git clone https://github.com/francismelojr/fraud-modeling.git
```

### 2. Navegue até o repositório

```bash
cd fraud-modeling
```

### 3. Configure a versão local do Python para a versão utilizada no projeto
```bash
pyenv install 3.11.5
pyenv local 3.11.5
```

### 4. Configure a versão correta do Python no Poetry
```bash
poetry env use 3.11.5
```

### 5. Instale as dependências do projeto
```bash
poetry install
```

### 6. Ative o ambiente virtual
```bash
poetry shell
```

Com isso, o projeto está devidamente configurado. Alguns comandos são pré configurados.
* Para treinar e exportar o modelo em um arquivo bin, o seguinte comando é utilizado

```bash
task train
```

* Para iniciar a API, utiliza-se o comando

```bash
task predict
```

* Para enviar uma requisição do tipo POST como teste:
```bash
task  test
```

## Iniciar API com Docker

Para inicializar a API em um container Docker, é necessário rodar os seguintes comandos

### Construa a imagem a partir do Dockerfile

```bash
docker build -t fraud-modeling
```

### Inicialize o container

```bash
docker run --rm -p 9696:9696 fraud-predict
```

### Novamente, podemos testar a API com o seguinte comando

```bash
task test
```

## Próximos passos

- [X] Rodar a API em um container Docker

- [ ] Realizar o deploy do projeto na nuvem, utilizando a imagem Docker e serviços AWS.

- [ ] Criar módulo de testes

- [ ] Criar workflow de CI