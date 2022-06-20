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
 		   - Led a team of three engineers to implement from scratch and successfully launch the firm’s first Customer Service System, which provides product consultation and complaint handling services to customers through the hotline, WeChat and intelligent assistant.
        - Designed micro-services architecture based on Spring boot and Dubbo, using MySQL and Redis Cluster as high-performance database, Kafka as message queue, Zookeeper for configuration.
        - Using Jenkins, Maven and Gitlab as CI/CD pipeline. Using ACK (Alibaba Cloud Container Service for Kubernetes) as the deployment environment for scalability. Using Alibaba OSS as file storage.
        - Built WeCom microservice based on SDK to implement online customer service handling process.
        - The system is capable of guaranteeing 500 customer service agents to handle business online, with a daily customer reception of over 15k.

 


  - title: Software Engineer
    company: Bitauto Holdings Limited
    company_url: 'http://www.bitauto.com/'
    company_logo: yiche
    location: Beijing, China
    date_start: '2018-07-05'
    date_end: '2019-11-14'
    description: |2-
      Founded in 2000, Bitauto was listed on the New York Stock Exchange in 2010 and became a member of the Tencent family in November 2020 when it completed its privatization. As a leading automotive Internet company in China, Bitauto provides professional and rich Internet information and shopping guide services for Chinese auto users, and effective Internet marketing solutions for auto manufacturers and auto dealers. <img src="https://gczhao.me/image/experience/yiche.jpeg" /> 
     
      Responsibility: 
        - Participated in the development of the Data Quality System, which monitors data changes in the data warehouse in multiple dimensions according to preset rules, and pushes alerts to relevant groups.
        - Used Spring Boot and Spring Cloud as MicroService Framework. Using MySQL and Redis as high-performance database. Using Hive as the data collection source.
        - Data Quality System processes approximately 1,500 tasks per day, with a data coverage rate of 75 percent and a notification delivery rate of 98 percent.
        - Participated in the secondary development and maintenance of the big data platform based on CDH. Mainly responsible for Oozie and HUE.


design:
  columns: '2'
---
