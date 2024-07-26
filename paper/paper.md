---
title: 'Deep River: A Deep Learning Library for Data Streams'
tags:
  - Python
authors:
  - name: Cedric Kulbach
    orcid: 0000-0002-9363-4728
    corresponding: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Lucas Cazzonelli
    orcid: 0000-0003-2886-1219
    affiliation: "1"
  - name: Hoang-Ahn Ngo
    orcid: 0000-0002-7583-753X
    affiliation: "2"
  - name: Max Halford
    orcid: 0000-0003-1464-4520
  - affiliation: "2"
affiliations:
 - name: FZI Research Center for Information Technology, Karlsruhe, Germany
   index: 1
 - name: Paris Telecom Tech
   index: 2
date: 20.10.2023
bibliography: paper.bib

---

## Summary
Machine learning algorithms enhance and expedite decision-making processes based on available data. 
However, as data evolves over time, it becomes crucial to adapt machine learning (ML) systems incrementally to accommodate new data patterns.
This adaptation is achieved through online learning or continuous ML technologies. 
Although deep learning technologies have demonstrated outstanding performance on predefined datasets, their application to online, streaming, and continuous learning scenarios has been limited.

[`Deep-River`](https://github.com/online-ml/deep-river) is a Python package for deep learning on data streams. 
Built on top of [`river`](https://riverml.xyz/latest/) and [`PyTorch`](https://pytorch.org), it offers a unified API for both supervised and unsupervised learning. 
Additionally, it provides a suite of tools for preprocessing data streams and evaluating deep learning models.

# Statement of need

In today's rapidly evolving landscape, machine learning (ML) algorithms play a pivotal role in shaping decision-making processes based on available data. 
The acceleration facilitated by these algorithms necessitates constant adaptation to evolving data structures, as what is valid at one moment may no longer hold in the future. 
To address this imperative, adopting online learning and continuous ML technologies becomes paramount.
While deep learning technologies have demonstrated exceptional performance on static, predefined datasets, their application to dynamic and continuously evolving data streams remains underexplored. 
The absence of widespread integration of deep learning into online, streaming, and continuous learning scenarios hampers the full potential of these advanced algorithms in real-time decision-making [@kulbach2024retrospectivetutorialopportunitieschallenges].
The emergence of the [`Deep-River`](https://github.com/online-ml/deep-river) Python package fills a critical void in the field of deep learning on data streams. 
Leveraging the capabilities of [`river`](https://riverml.xyz/latest/)[@montiel2021river] and [`PyTorch`](https://pytorch.org), `Deep-River` offers a unified API for both supervised and unsupervised learning, providing a seamless bridge between cutting-edge deep learning techniques and the challenges posed by dynamic data streams. 
Moreover, the package equips practitioners with essential tools for data stream preprocessing and the evaluation of deep learning models in dynamic, real-time environments. 
This was already made use of in the context of Streaming Anomaly Detection [@cazzonelli2022detecting].
As the demand for effective and efficient adaptation of machine learning systems to evolving data structures continues to grow, the integration of `Deep-River` into the landscape becomes crucial. 
This package stands as a valuable asset, unlocking the potential for deep learning technologies to excel in online, streaming, and continuous learning scenarios. 
The need for such advancements is evident in the quest to harness the full power of machine learning in dynamically changing environments, ensuring our decision-making processes remain accurate, relevant, and agile in the face of evolving data landscapes.

# Features
[`Deep-River`](https://github.com/online-ml/deep-river) enables the usage of deep learning models for data streams. 
This means that deep learning models need to adapt to changes within the evolving data stream i.ex. the number of classes might change over time.
In addition to the integration of [`PyTorch`](https://pytorch.org) into the [`river`](https://riverml.xyz/latest/)[@montiel2021river] this package offers additional data stream specific functionalities such as class incremental learning or specific optimizers for data streams.

## Compatibility
[`Deep-River`](https://github.com/online-ml/deep-river) is built on the unified application programming interface (API) of [`river`](https://riverml.xyz/latest/)[@montiel2021river] that seamlessly integrates both supervised and unsupervised learning techniques.
Further, it integrates the huge functionality of [`PyTorch`](https://pytorch.org) for deep learning such as using GPU acceleration and a broad range of architectures.
This unified approach simplifies the development process and facilitates a cohesive workflow for practitioners working with dynamic data streams.
Leveraging the capabilities of the well-established [`river`](https://riverml.xyz/latest/)[@montiel2021river] library and the powerful [`PyTorch`](https://pytorch.org) framework, `Deep-River` combines the strengths of these technologies to deliver a robust and flexible platform for deep learning on data streams. 
This foundation ensures reliability, scalability, and compatibility with state-of-the-art machine learning methodologies.
It provides comprehensive [documentation](https://online-ml.github.io/deep-river/) to guide users through the installation, implementation, and customization processes. Additionally, a supportive community ensures that users have access to resources, discussions, and assistance, fostering a collaborative environment for continuous improvement and knowledge sharing.

## Adaptivity
[`Deep-River`](https://github.com/online-ml/deep-river) is specifically designed to cater to the requirements of online learning scenarios. 
It enables continuous adaptation to evolving data by supporting incremental updates and learning from new observations in real time, a critical feature for applications where data arrives sequentially.
Further, it enables the model to adapt to changes in the number of classes over time for classification tasks.
It equips practitioners with tools for evaluating the performance of deep learning models on data streams. This feature is crucial for ensuring the reliability and effectiveness of models in real-time applications, enabling users to monitor and fine-tune their models as the data evolves.

# Architecture
The `deep_river` library is structured around various types of estimators for anomaly detection, classification, and regression. 
Each category contains several specialized classes that inherit from more general base classes, forming a hierarchical structure.
In anomaly detection, the base class `AnomalyScaler` has derived classes `AnomalyMeanScaler`, `AnomalyMinMaxScaler`, and `AnomalyStandardScaler`. 
Additionally, the `Autoencoder` class, which inherits from `DeepEstimator`, has a specialized subclass called `ProbabilityWeightedAutoencoder`. 
The `RollingAutoencoder` class inherits from `RollingDeepEstimator`.

For classification, the base class `Classifier` inherits from `DeepEstimator`. 
Derived from `Classifier` are specific classes like `LogisticRegression` and `MultiLayerPerceptron`. 
The `RollingClassifier` class inherits from both `RollingDeepEstimator` and `Classifier`.

In regression, the base class `Regressor` inherits from `DeepEstimator`. 
Specific regression classes like `LinearRegression` and `MultiLayerPerceptron` inherit from `Regressor`. 
The `MultiTargetRegressor` also inherits from `DeepEstimator`. 
The `RollingRegressor` class inherits from both `RollingDeepEstimator` and `Regressor`.

![Architecture of Deep-River\label{fig:Architecture}](classes.png){width=100%}

Overall, the library is organized to provide a flexible and hierarchical framework for different types of machine learning tasks, with a clear inheritance structure connecting more specific implementations to their base classes.

# Acknowledgements

This work was carried out with the support of the Research Center for Information Technology in Karlsruhe, Germany.

# References