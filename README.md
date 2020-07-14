# Opoca

Opoca library aims to drastically speed up producing proof of concepts (PoC) for machine learning projects. 

We define proof of concept as a small, quick and (not) dirty projects that results in:
- exploratory data analysis
- log of experiments along with models
- deployable best model
- demo (jupyter notebook/streamlit app etc.)
- short report including results analysis

There are several challenges that ML Engineer faces given a task to build new PoC project:
* it's not easy to track and reproduce experiments
* it's not easy to version and share data
* it's not easy to schedule jobs and not burn much money on training
* there's a lot of code that can be reused between different PoCs such as:
    * training logic for similar problems
    * evaluation logic
    * plotting
    * hyperparameters search
    * generic feature engineering transformations
    
Those are just few and a list is not complete without a doubt.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of [poetry](https://github.com/python-poetry/poetry)

## Installing Opoca

Opoca is installable from PyPi by executing:

```shell script
pip install opoca
```

One may also use docker to build image:

```shell script
docker build -t opoca -f Dockerfile .
```

And run bash session interactively by executing:

```shell script
docker run -it --rm -v $PWD:/home -w /home opoca bash
```

## Contributing to Opoca
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to `Opoca`, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:
* [@plazowicz](https://github.com/plazowicz)
* [@pedrito87](https://github.com/pedrito87)
* [@mjmikulski](https://github.com/mjmikulski)
* [@AdrianMaciej](https://github.com/AdrianMaciej)
* [@dkosowski87](https://github.com/dkosowski87)

## Contact

If you want to contact me you can reach me at <ml-team@netguru.com>.

## License

This project uses the following license: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
