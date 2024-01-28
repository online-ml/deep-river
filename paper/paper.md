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

# Summary
Machine learning algorithms have become indispensable in today's world. 
They support and accelerate the way we make decisions based on the data at hand. 
This acceleration means that data structures that were valid at one moment could no longer be valid in the future. 
With these changing data structures, it is necessary to adapt machine learning (ML) systems incrementally to the new data. 
This is done with the use of online learning or continuous ML technologies. 
While deep learning technologies have shown exceptional performance on predefined datasets, they have not been widely applied to online, streaming, and continuous learning.
[`Deep-River`](https://github.com/online-ml/deep-river) is a python package for deep learning on data streams. 
It is built on top of [`river`](https://riverml.xyz/latest/) and [`PyTorch`](https://pytorch.org) and provides a unified API for both supervised and unsupervised learning. 
It also provides a set of tools for data stream preprocessing and evaluation of deep learning models.

# Statement of need

In today's rapidly evolving landscape, machine learning (ML) algorithms play a pivotal role in shaping decision-making processes based on available data. The acceleration facilitated by these algorithms necessitates constant adaptation to evolving data structures, as what is valid at one moment may no longer hold true in the future. To address this imperative, the adoption of online learning and continuous ML technologies becomes paramount.
While deep learning technologies have demonstrated exceptional performance on static, predefined datasets, their application to dynamic and continuously evolving data streams remains underexplored. The absence of widespread integration of deep learning into online, streaming, and continuous learning scenarios hampers the full potential of these advanced algorithms in real-time decision-making.
The emergence of the [`Deep-River`](https://github.com/online-ml/deep-river) Python package fills a critical void in the field of deep learning on data streams. Leveraging the capabilities of [`river`](https://riverml.xyz/latest/) and [`PyTorch`](https://pytorch.org), `Deep-River` offers a unified API for both supervised and unsupervised learning, providing a seamless bridge between cutting-edge deep learning techniques and the challenges posed by dynamic data streams. Moreover, the package equips practitioners with essential tools for data stream preprocessing and the evaluation of deep learning models in dynamic, real-time environments.
As the demand for effective and efficient adaptation of machine learning systems to evolving data structures continues to grow, the integration of `Deep-River` into the landscape becomes crucial. This package stands as a valuable asset, unlocking the potential for deep learning technologies to excel in online, streaming, and continuous learning scenarios. The need for such advancements is evident in the quest to harness the full power of machine learning in dynamically changing environments, ensuring our decision-making processes remain accurate, relevant, and agile in the face of evolving data landscapes.
# Features

Features of the `Deep-River` Package:

1. **Unified API for Supervised and Unsupervised Learning:**
   - `Deep-River` provides a unified application programming interface (API) that seamlessly integrates both supervised and unsupervised learning techniques. This unified approach simplifies the development process and facilitates a cohesive workflow for practitioners working with dynamic data streams.

2. **Built on Top of `river` and `PyTorch`:**
   - Leveraging the capabilities of the well-established [`river`](https://riverml.xyz/latest/) library and the powerful [`PyTorch`](https://pytorch.org) framework, `Deep-River` combines the strengths of these technologies to deliver a robust and flexible platform for deep learning on data streams. This foundation ensures reliability, scalability, and compatibility with state-of-the-art machine learning methodologies.

3. **Support for Online Learning:**
   - `Deep-River` is specifically designed to cater to the requirements of online learning scenarios. It enables continuous adaptation to evolving data by supporting incremental updates and learning from new observations in real-time, a critical feature for applications where data arrives sequentially.

4. **Deep Learning on Data Streams:**
   - Addressing the gap in the application of deep learning to dynamic data streams, `Deep-River` focuses on optimizing deep learning models for real-time, streaming, and continuous learning. This feature extends the applicability of deep learning algorithms to scenarios where data is not static but evolves over time.

5. **Model Evaluation Capabilities:**
   - `Deep-River` equips practitioners with tools for evaluating the performance of deep learning models on data streams. This feature is crucial for ensuring the reliability and effectiveness of models in real-time applications, enabling users to monitor and fine-tune their models as the data evolves.

6. **Open Source and Extensible:**
   - Being an open-source Python package, `Deep-River` encourages collaboration and contribution from the community. Its modular and extensible architecture allows users to customize and extend functionalities, promoting the development of tailored solutions for diverse applications in the realm of online and continuous learning.

7. **Documentation and Community Support:**
   - `Deep-River` provides comprehensive [documentation](https://online-ml.github.io/deep-river/) to guide users through the installation, implementation, and customization processes. Additionally, a supportive community ensures that users have access to resources, discussions, and assistance, fostering a collaborative environment for continuous improvement and knowledge sharing.

# Acknowledgements

This work was carried out with the support of the Research Center for Information Technology in Karlsruhe, Germany.

# References