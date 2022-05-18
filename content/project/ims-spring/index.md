---
title: IMS
summary: An Inventory Management System developed with SpringBoot, MySQL, Mybatis, Mybatis-Plus, and Redis.
tags:
  - Spring Boot
date: '2021-12-10T00:00:00Z'

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption:
  focal_point: Smart

links:
  - icon: github
    icon_pack: fab
    name: Follow
    url: https://github.com/zhaoguanchen
 
url_code: 'https://github.com/zhaoguanchen/IMS'
url_pdf: '/uploads/pdf/report.pdf'
url_slides: '/uploads/slides/ims.pptx'
url_video: 'https://youtu.be/03hlfeOkhHE'


# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ''
---

# IMS - an inventory management system

## Description

The main purpose to develop this project is that practice the tools such as SpringBoot, MySQL, Mybatis, Mybatis-Plus, and Redis.

**SpringBoot**

Spring Boot is an open-source Java-based framework used to create a micro Service.

**MySQL**

MySQL is an [open-source](https://en.wikipedia.org/wiki/Open-source_software) [relational database management system](https://en.wikipedia.org/wiki/Relational_database_management_system) (RDBMS).

**Mybatis**

MyBatis is a first-class persistence framework with support for custom SQL, stored procedures, and advanced mappings. MyBatis eliminates almost all of the JDBC code and manual setting of parameters and retrieval of results. MyBatis can use simple XML or Annotations for configuration and map primitives, Map interfaces, and Java POJOs (Plain Old Java Objects) to database records.

**mybatis-plus**

Mybatis plus is an enhanced tool of Mybatis. After using Mybatis plus, we can not only use the unique functions of Mybatis-plus but also use the native functions of Mybatis normally. 

**Redis**

Redis is an open-source, in-memory data store used by millions of developers as a database, cache, streaming engine, and message broker.

# Features

**Front-end and back-end separation architecture**

Front-end: **React**

Back-end: **Spring-Boot**

In the decoupled web architecture, frontend and backend code are separate, with no shared resources. The two codebases communicate, but each has its server instance. The backend application serves data via an API (application programming interface) using a framework such as JSON (JavaScript Object Notation). A decoupled approach is advantageous for facilitating changes, enabling individual services and components to be independently scaled up or down or removed entirely without disrupting the entire application. Additionally, decoupling allows frontend and backend developers to optimize their portions without fear of how their work impacts the rest of the system. Developers in general prefer the decoupled approach as it tends to remove production bottlenecks, simplify testing and make for a more easily reusable backend codebase.



**Authentication**

After the user logs in, the authentication token will be returned to the user, which is required to be carried in the header of subsequent requests. The token is stored in **Redis**, the key is the token, value is user information.

**Application of Cache Service**

We have access to Redis to provide us with the caching service.

**Https**

Our requests are all made through HTTPS, in which we use the domain name and the certificate service provided by AWS.

**Soap and Restful**

We combine restful service with soap service. The framework is spring boot and spring WS.

**Data Encryption**

We use `md5 + salt` to encrypt sensitive information, such as passwords.



# Architectural Design

**Major modules**

We have 4 modules:

- **Login**: contains login, register, and logout.

- **User**: contains add, update, delete, and page list with search.

- **Inventory**: contains add, update, delete, and page list with search.

- **Subscribe**: contains add, update, cancel, and page list with search.

- **Appointment**: contains add, update, cancel and page list with search.





 

**Service Architecture**
**Load balance**: The backend service is deployed on [AWS Elastic Beanstalk](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/Welcome.html), so we config load balance with AWS Elastic Beanstalk. We have two pods to process the request.

**Redis**: The Redis server is installed on AWS EC2, we use Redis to save token and user information.

**RDS**: we use Google RDS - MySQL as our data storage.

**AWS Email**: we use AWS Email Service to send emails to users

**Twilio**: we use Twilio to send text messages to users.

**Timer**: The timer is used to realize the function of sending reminders regularly.

![img](https://lh3.googleusercontent.com/T9Txft-hDJvWYdvxfODHII3_bvlN6Pb3DLBkR0u4BnAFldexHvgmRw_RNbdPUsMhF8jxv86cX9Wjd07jeLY5Qw8TDB04VWLe_NCf9lZIbe-OnvTbRoC-Stc7DepNV3YxDzXjvUkhZfbLoDAnEQ)

**Code Architecture**
For the design of code, we use Spring Boot Framework and Spring WS. The layout is below.

**Controller**: the controller is the portal of the service. The main work is to check and parse the parameter and check the user token for security.

**Service**: The service is the main part of handling logical transactions.

**Mapper**: The mapper is responsible for the interaction with the database. His dependency is mybatis framework.

**Util**: Util contains a lot of tool classes, which is a lower level.
![img](https://lh4.googleusercontent.com/FD6OmIYorXx2cToif8MGVnxqrF_kOOT3-PDSeAV4yPGqi5640fOjhjhpT6IrwCp0o7RRqQdS8yCQGZfhda8GVLAXJBmZehVUdwHSYCn0dW0fO6CyreqfoQm_dodhx4DODc4o9gVtzy9cfYdhSA)



**Sample Diagram**
It is the login and register module. 
![img](https://lh3.googleusercontent.com/ftNyBZwWB2xRb0a1eBQOgPVhspTD7vlyZhS8LEhbaccywddOszQOshxOEIbkR4T6kT6aOBjkuLPSGNftiq58KQ3Pr7sjf9QVZZxu7Q-7-FGrYuxpRXkw0muyhaF41MtyrbBhbFA5sGiowth6UA)

**Class Architecture**

There is a snapshot diagram of the classes, which contains most of the classes and their connection.
![img](https://lh5.googleusercontent.com/nmHiL6Uq6GLQ6nog9z69OQRyUAbriZR2hYPQy4pj3EsNOO8iKUjh9secZd6aWSCpBupqiT_kYt_Q1QixgomCUPllcRs67bAiBh3FLJL231HupiiKdBa1Rx8BGldk6p4v_Md4fD2VLI6VpP2IfQ)

**Component Diagram**
The component diagram depicts how components are wired together to form larger components or software systems.

There are three components: Web service, Warehouse and Subscribe center.![img](https://lh4.googleusercontent.com/TTgeRWaqBZkysrAjpw5d0QG0jw1V94Dc6-0jUf0hedKkUDXkIgGLUbbiaTAiK8-oGvwRRm2nGnONxQzg3HRhjDD3TkEuyQqH6k-kapOkP3MMOjWKfosXZnAOje5aB6hltqLr1dQijwmFq1oPmQ)				 Component Diagram

**Sequence Diagram**
The sequence diagram shows such a process:

1. The user logs in or registers the system. 
2. Users browse the cultural relics in the museum collection and find them through search.
3. The user initiates the attention subscription for the cultural relic. 
4. The system sends a reminder message (exhibition time and place) to the user through SMS or e-mail.![img](https://lh3.googleusercontent.com/Ra5aBS8MrUUaAzpFlREarhcMRW7VLupDfLvsOfFpE9CzKaJvnmMmF33aR7WaVQWOhyD5FBkYruwj4qzp1fkxZW5WIn7JIkWebC3CoFrH4UyOU9brOhSLtMQAR7Vi5_ux306-URu9K5oZB8ahBg) 



# Web API Design



Please click the link to review the Web API Design.

[Web API Design](https://documenter.getpostman.com/view/3290385/UVR4M9nP)

# Deploy

We tried multiple platforms for deployment.

**Back-end**: AWS Elastic Beanstalk and AWS EC2.

**Front-end**: AWS Amplify and Google App Engine.

**Database**: RDS on Google Cloud Platform and RDS on AWS.

**Redis**: AWS EC2.



# Getting Started

This is an example of how you may give instructions on setting up the project locally. To get a local copy up and running follow these simple example steps.

#### Installation

Install Java 1.8 at https://www.oracle.com/java/technologies/downloads/

Install Maven 3.8.4. at https://maven.apache.org/install.html

#### run

run maven command at project directory`/ims`  

```shell
mvn install
```

run maven command at project directory `/ims`

```bash
mvn spring-boot:run
```

#### code structure

the root is `com.gc.ims`

the main class is `ImsApplication.class`

`application.properties` contains configuration, such as the database connection information.

`sql/sql.sql` contains the DML for the database table and data.

#### platform and IDE

the IDE is `intelliJ IDEA`.

# Demo

[Click](https://youtu.be/03hlfeOkhHE) to watch the demo video.

# Other

email guanchenzhao@gmail.com for more information.

