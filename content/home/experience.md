---
# An instance of the Experience widget.
# Documentation: https://wowchemy.com/docs/page-builder/
widget: experience

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 30

title: Experience
subtitle:

# Date format for experience
#   Refer to https://wowchemy.com/docs/customization/#date-format
date_format: Jan 2006

# Experiences.
#   Add/remove as many `experience` items below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
experience:
  - title: Senior Software Engineer
    company: Yuanbao
    company_url: 'https://www.yuanbaobaoxian.cn/'
    company_logo: yuanbao
    location: Beijing, China
    date_start: '2019-11-15'
    date_end: '2021-08-20'
    description: |2-
      Yuanbao is an Internet insurance platform launched in 2020. As of May 2021, Yuanbao has accumulated millions of paying users and its business covers more than 90% of China. At the same time, it has completed the C round of financing of nearly RMB 1 billion. <img src="https://gczhao.me/image/experience/yuanbao_big.png" /> 
       
      Responsibility: 
        - Led a team consisting of 3 engineers to implement from scratch and successfully launch the firm’s first Customer Service System – CSS, which provides customers with product consultation and complaint handling services through the hotline, WeChat and AI assistants. 
        - Designed architecture of native service mesh cloud app on top of Golang backend services, using MySQL and Redis as high-performance database, Prow, Bazel as CI/CD, Hive, ClickHouse as data statistics and AWS by K8s + Istio as deployment environment.
        - Themis ecosystem managed to accumulate 200 million users within 10 days of its release to the public, with delivery rate above 97%, daily peak value over  5000k and online connections averaged 10k-15k QPS.

 


  - title: Software Engineer
    company: Bitauto Holdings Limited
    company_url: 'http://www.bitauto.com/'
    company_logo: yiche
    location: Beijing, China
    date_start: '2018-07-05'
    date_end: '2019-11-14'
    description: |2-       
     Responsibility: 
        - Participated in the development of Data Quality System, which  monitors data changes in the data warehouse in multiple dimensions according to preset rules, and pushes alerts to relevant groups. 
        - Used Spring Boot and Spring Cloud as MicroService Framework.
        - Participated in the secondary development and maintenance of the big data platform based on [CDH](https://www.cloudera.com/products/open-source/apache-hadoop/key-cdh-components.html). Mainly responsible for Oozie and HUE.


design:
  columns: '2'
---
