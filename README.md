# Phase 4 Project Description

Final phase down -- you're absolutely crushing it! You've made it all the way through one of the toughest phases of this course. You must have an amazing brain in your head!

<img src='https://raw.githubusercontent.com/learn-co-curriculum/dsc-phase-4-project-v2-3/main/images/brain.gif'>

You have one last project to complete before the Capstone!

In this project description, we will cover:

* Project Overview
* Deliverables
* Grading
* Getting Started

## Project Overview

For this project, you will engage in an **advanced supervised modeling process** from start to finish, solving either a classification or a regression problem using an advanced dataset.

### Business Problem and Data

Similar to the Phase 3 project, you are responsible for choosing a dataset as well as defining a stakeholder and business problem. In addition to these choices, you can choose between any of the four advanced supervised modeling techniques covered in Phase 4:

- Time Series Modeling
- Recommendation System
- Image Classification with Deep Learning
- Natural Language Processing

For complete details, see [Phase 4 Project - Choosing a Dataset](https://github.com/learn-co-curriculum/dsc-phase-4-choosing-a-dataset).

### Key Points

### Advanced Data Types and Modeling

The purpose of this project is to demonstrate that you have mastered the basics of some type of modeling technique beyond the techniques introduced in Phase 3. This is your change to tailor your work to a **data science audience** in particular, with a clear notebook narrative that illustrates your process. The resulting presentation slides will be substantially similar to a Phase 3 presentation, but someone reading your notebook should be able to see your grasp of the specific advanced modeling technique.

### Validation Strategy

A **validation strategy** means a strategy to demonstrate that your model will actually perform well on unseen data. In Phase 3 this was relatively straightforward to accomplish with the `train_test_split` function from scikit-learn. This may or may not be appropriate for the project you select. Make sure that you are thinking about this strategy from the start and incorporating it into your notebook narrative.

### Choosing a Dataset

We've given you a lot of choices - don't get stuck spending too much time choosing which project to do. Give yourself a firm time limit for picking a project (e.g. 2 hours) so you can get on with making something great. Don't worry about picking the perfect project - remember that you will get to do a new, larger Capstone project very soon!

## The Deliverables

There are three deliverables for this project:

* A **non-technical presentation**
* A **Jupyter Notebook**
* A **GitHub repository**

The checklist of requirements is the same for Phase 4 as it was in Phase 3. This will also be the checklist for Capstone!

### Non-Technical Presentation

The non-technical presentation should be very similar to the presentation you gave in Phase 3. You can feel free to mention the specific models and metrics you used, but make sure you translate everything for an audience who is not familiar with data science.

### Jupyter Notebook

The notebook will have the same checklist elements as in Phase 3. However, **this time around the Communication rubric element will focus on the technical notebook**. A data science professional reading your notebook should be able to understand all of your data preparation and modeling decisions.

### GitHub Repository

The GitHub repository should also be very similar to the Phase 3 repository.

The main additional element to consider is **reproducibility**, since many of the dataset options are too large to be saved directly in a GitHub repository. Make sure you include clear instructions for how someone would reproduce your modeling process, potentially including any scripts you used to organize data into directories.

## Grading

***To pass this project, you must pass each rubric objective.*** The project rubric objectives for Phase 4 are:

1. Attention to Detail
2. Advanced ML Communication
3. Advanced Data Preparation
4. Advanced ML Modeling

### Attention to Detail

Once again, the Attention to Detail standard has increased. ***In Phase 4, you need to complete 90% (9 out of 10) or more of the checklist elements in order to pass the Attention to Detail objective.***

**NOTE THAT THE PASSING BAR IS HIGHER IN PHASE 4 THAN IT WAS IN PHASE 3!**

#### Exceeds Objective

100% of the project checklist items are complete

#### Meets Objective (Passing Bar)

90% of the project checklist items are complete

#### Approaching Objective

80% of the project checklist items are complete

#### Does Not Meet Objective

70% or fewer of the project checklist items are complete

### Advanced ML Communication

Once again, you are expected to communicate the results of an ML modeling process. Just like in Phase 3, we are looking for *rationale, results, limitations, and recommendations.*

In Phase 4, the emphasis is on the **Jupyter Notebook**. The notebook should include a **summary** at the beginning that briefly and accurately describes your process. The summary should be approximately 250 words -- about the size of a research paper abstract.

Summary elements:

* Business and data understanding: *what kind of data are you using, and what makes it well-suited for the business problem?*
  * You do not need to include any data visualizations in your summary, but consider including relevant descriptive statistics
* Data preparation: *why did you choose the data preparation steps that you did, and what was the result?*
  * This should be specific to the kind of data you are working with. For example, if you are doing an NLP project, what did you decide to do with stopwords?
  * Be sure to list the packages/libraries used to prepare the data, and why
* Modeling: *what modeling package(s) did you use, which model(s) within the package(s), and what tuning steps did you take?*
  * For some projects there may be only one applicable package; you should still briefly explain why this was the appropriate choice
  * For neural networks projects, be sure to describe your model architecture choices
* Evaluation: *how well did your final model perform?*
  * Include one or more relevant metrics
  * Be sure to briefly describe your validation approach

#### Exceeds Objective

Communicates advanced modeling summary as well as a narrative throughout the notebook text that demonstrates mastery of an advanced topic

> Decisions should be justified and outcomes evaluated in Markdown throughout the notebook

#### Meets Objective (Passing Bar)

Successfully communicates a summary of an advanced modeling technique including business and data understanding, data preparation, modeling, and evaluation

> It is possible to meet this bar with just a summary and not a narrative throughout the notebook, so long as the steps taken are justifiable and free of major errors. See the Approaching Objective section for an explanation of what a "major error" means.

#### Approaching Objective

Communicates advanced modeling summary with at least one major error

> A major error means that one of the required elements of the summary was missing, or some aspect of the communication was fundamentally incorrect. For example, if you stated that you performed "deep learning" when you actually used `CountVectorizer` and `MultinomialNB` from scikit-learn, that would be a major error. Another example would be if you described a regression task as a classification task, or if you described supervised learning as unsupervised learning.

#### Does Not Meet Objective

Does not communicate advanced modeling summary

> Markdown headings and occasional narrative text throughout the notebook are not sufficient in this phase, even if they were in previous phases. You need to include a summary at the beginning of your notebook.

### Advanced Data Preparation

Once again, this objective is very similar to Phase 3, although the complexity has increased.

#### Exceeds Objective

Goes above and beyond with data preparation, such as feature engineering, using pipelines, or using unsupervised techniques

> Supervised learning is the core of this project, but feel free to use unsupervised techniques for data analysis or preparation

#### Meets Objective (Passing Bar)

Successfully prepares data for modeling, using at least one Python package other than scikit-learn

> Your final model does not need to use anything other than scikit-learn, but you should explore other tools during your modeling process

#### Approaching Objective

Prepares some data successfully, but has at least one major error

#### Does Not Meet Objective

Does not prepare data for modeling

### Advanced ML Modeling

This is your real opportunity to flex those new Phase 4 skills!

#### Exceeds Objective

Goes above and beyond in the modeling process, such as using models from multiple different packages or model explainability tools

> You are encouraged but not required to use models from multiple different packages. The feasibility of this depends on your choice of project. For time series, this might mean trying both StatsModels and Prophet. For image classification, this might mean using TensorFlow with and without transfer learning.

> See [this book chapter](https://christophm.github.io/interpretable-ml-book/lime.html) for an introduction to LIME for model explainability

#### Meets Objective (Passing Bar)

Successfully builds and evaluates multiple models using an appropriate model validation technique

> As noted previously, the `train_test_split` from scikit-learn may or may not be appropriate for your modeling task. Be sure to investigate appropriate techniques so you are confident in the performance of your final model on unseen data

#### Approaching Objective

Builds multiple models with at least one major error

#### Does Not Meet Objective

Does not build multiple models

## Getting Started

Please start by reviewing the contents of this project description. If you have any questions, please ask your instructor ASAP.

Next, select a dataset and a business problem. Recall that more information is available in [Phase 4 Project - Choosing a Dataset](https://github.com/learn-co-curriculum/dsc-phase-4-choosing-a-dataset).

To get started with project development, create a new repository on GitHub. For this project, we recommend that you do not fork the template repository, but rather that you make a new repository from scratch, starting by going to [github.com/new](https://github.com/new).

## Summary

This project is your chance to show off your data science prowess with some advanced machine learning algorithms. Now that you've gone through all of the core course content, we're excited to see what you are able to do!
