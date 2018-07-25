# Deploy spark as cluster-mode on Nectar Cloud

- when you create the cloud, make sure the security setting include the following:
the folowwing setting needed to be include in the securaity group, when you use port 90000, it is quite necessary

| Direction | Ether Type | IP Protocol | Port Range | Remote IP Prefix |
| --- | --- | --- | --- | --- |
| Egress | IPv4  | Any | Any | 0.0.0.0/0 |
| Egress | IPv6  | Any | Any | ::/0 |
| Ingress | IPv4  | Any | Any | 0.0.0.0/0 |

| Direction | Ether Type | IP Protocol | Port Range | Remote IP Prefix |
| --- | --- | --- | --- | --- |
| Egress | IPv4  | Any | Any | 0.0.0.0/0 |
| Egress | IPv6  | Any | Any | ::/0 |
| Ingress | IPv4  | TCP | 22(SSH) | 0.0.0.0/0 |
| Ingress | IPv4  | TCP | 80(HTTP) | 0.0.0.0/0 |
| Ingress | IPv4  | TCP | 9000 | 0.0.0.0/0 |
| Ingress | IPv4  | TCP | 7077 | 0.0.0.0/0 |

## Enter nodes ip
sudo nano /etc/hosts

    master  XXXXXXXX
    slave1  XXXXXXXX
    slave2  XXXXXXXX

## Edit bashrc

sudo nano ~/.bashrc

    #HADOOP VARIABLES START
    export JAVA_HOME=/home/ubuntu/spark/jdk1.7.0_75
    export HADOOP_INSTALL=/home/ubuntu/spark/hadoop-2.6.0
    export PATH=$PATH:$HADOOP_INSTALL/bin
    export PATH=$PATH:$HADOOP_INSTALL/sbin
    export HADOOP_MAPRED_HOME=$HADOOP_INSTALL
    export HADOOP_COMMON_HOME=$HADOOP_INSTALL
    export HADOOP_HDFS_HOME=$HADOOP_INSTALL
    export YARN_HOME=$HADOOP_INSTALL
    export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_INSTALL/lib/native
    export HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_INSTALL/lib"
    export LD_LIBRARY_PATH=$HADOOP_INSTALL/lib/native
    #HADOOP VARIABLES END

source ~/.bashrc

## add user
- NOT KNOWING ITS NECCESSARY</li>

sudo useradd -m hadoop -s /bin/bash<br />
sudo passwd hadoop<br />
sudo adduser hadoop sudo

## Working directory
sudo mkdir spark

## install java
sudo mv /home/ubunt/jdk-7u75-linux-x64.tar.gz spark/<br />
cd spark/<br />
sudo tar -zxvf jdk-7u75-linux-x64.tar.gz<br />
sudo nano /etc/profile

    export WORK_SPACE=/home/ubuntu/spark/          
    export JAVA_HOME=$WORK_SPACE/jdk1.7.0_75
    export JRE_HOME=/home/spark/work/jdk1.7.0_75/jre
    export PATH=$JAVA_HOME/bin:$JAVA_HOME/jre/bin:$PATH
    export CLASSPATH=$CLASSPATH:.:$JAVA_HOME/lib:$JAVA_HOME/jre/lib

source /etc/profile

## install scala
sudo wget www.scala-lang.org/files/archive/scala-2.10.4.tgz<br />
sudo tar -zxvf scala-2.10.4.tgz<br />
sudo apt-get update<br />
sudo nano /etc/profile

    export SCALA_HOME=$WORK_SPACE/scala-2.10.4
    export PATH=$PATH:$SCALA_HOME/bin

source /etc/profile

## ssh
sudo apt-get install ssh<br />
sudo apt-get install openssh-server<br />
ssh-keygen -t rsa<br />
- dont't use sudo to generate the key
- Enter all the way
- Do this on all slaves
- send key to the master</li>

sudo scp /home/ubuntu/.ssh/id_rsa.pub ubuntu@master:/home/ubuntu/.ssh/id_rsa.pub.slave1<br />
- on master, add all key to the authorized key</li>

sudo cat /home/ubuntu/.ssh/id_rsa.pub* >> /home/ubuntu/.ssh/authorized_keys<br />
- send authorized_keys to all slaves</li>

sudo scp /home/ubuntu/.ssh/authorized_keys spark@slave1:/home/ubuntu/.ssh/<br />

## install hadoop
cd spark<br />
sudo wget http://mirrors.sonic.net/apache/hadoop/common/hadoop-2.6.0/hadoop-2.6.0.tar.gz<br />
sudo tar -zxvf hadoop-2.6.0.tar.gz<br />
cd hadoop-2.6.0/etc/hadoop<br />
sudo nano hadoop-env.sh

    export JAVA_HOME=/home/ubuntu/spark/jdk1.7.0_75

sudo nano yarn-env.sh 

    export JAVA_HOME=/home/ubuntu/spark/jdk1.7.0_75

sudo nano slaves

    slave1
    slave2

sudo nano core-site.xml

        <property>
            <name>fs.defaultFS</name>
            <value>hdfs://master:9000/</value>
        </property>
        <property>
            <name>hadoop.tmp.dir</name>
            <value>file:/home/ubuntu/spark/hadoop-2.6.0/tmp</value>
        </property>

sudo nano hdfs-site.xml

        <property>
            <name>dfs.namenode.secondary.http-address</name>
            <value>slave1:9001</value>
        </property>
        <property>
            <name>dfs.namenode.name.dir</name>
            <value>file:/home/ubuntu/spark/hadoop-2.6.0/dfs/name</value>
        </property>
        <property>
            <name>dfs.datanode.data.dir</name>
            <value>file:/home/ubuntu/spark/hadoop-2.6.0/dfs/data</value>
        </property>
        <property>
            <name>dfs.replication</name>
            <value>3</value>
        </property>

sudo nano yarn-site.xml

        <property>
            <name>yarn.nodemanager.aux-services</name>
            <value>mapreduce_shuffle</value>
        </property>
        <property>
            <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
            <value>org.apache.hadoop.mapred.ShuffleHandler</value>
        </property>
        <property>
            <name>yarn.resourcemanager.address</name>
            <value>master:8032</value>
        </property>
        <property>
            <name>yarn.resourcemanager.scheduler.address</name>
            <value>master:8030</value>
        </property>
        <property>
            <name>yarn.resourcemanager.resource-tracker.address</name>
            <value>master:8035</value>
        </property>
        <property>
            <name>yarn.resourcemanager.admin.address</name>
            <value>master:8033</value>
        </property>
        <property>
            <name>yarn.resourcemanager.webapp.address</name>
            <value>master:8088</value>
        </property>

sudo cp mapred-site.xml.template mapred-site.xml<br />
sudo nano mapred-site.xml


        <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
        </property>

## start hadoop at master
cd /home/ubuntu/spark/hadoop-2.6.0<br />
sudo chmod a+w /home/ubuntu/spark/hadoop-2.6.0/<br />
- disable ipv6</li>

sudo nano /etc/sysctl.conf

    net.ipv6.conf.all.disable_ipv6 = 1
    net.ipv6.conf.default.disable_ipv6 = 1
    net.ipv6.conf.lo.disable_ipv6 = 1

sudo sysctl -p<br />
cat /proc/sys/net/ipv6/conf/all/disable_ipv6

bin/hadoop namenode -format<br />
sbin/start-dfs.sh <br />
sbin/start-yarn.shyarn<br />
- test the datanode

bin/hdfs dfsadmin -report

## start spark
sudo mv /home/ubunt/spark-2.0.2-bin-hadoop2.6.tgz /home/ubuntu/spark<br />
cd spark<br />
sudo tar -zxvf spark-2.0.2-bin-hadoop2.6.tgz<br />
cd spark-2.0.2-bin-hadoop2.6/conf/<br />
sudo cp spark-env.sh.template spark-env.sh<br />
- configurate the spark properties</li>

sudo nano spark-env.sh

    export SCALA_HOME=/home/ubuntu/spark/spark-2.0.2-bin-hadoop2.6
    export JAVA_HOME=/home/ubuntu/spark/jdk1.7.0_75
    export HADOOP_HOME=/home/ubuntu/spark/hadoop-2.6.0
    export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
    SPARK_MASTER_IP=master
    SPARK_LOCAL_DIRS=/home/ubuntu/spark/spark-2.0.2-bin-hadoop2.6
    SPARK_DRIVER_MEMORY=4G

sudo chmod a+w /home/ubuntu/spark/spark-2.0.2-bin-hadoop2.6
sudo cp slaves.template slaves<br />
sudo nano slaves<br />

    master
    slave1
    slave2

../sbin/start-all.sh<br />


## Test
- ALS example</li>

    ./bin/spark-submit
         --master spark://master:7077
         examples/src/main/python/als.py 100 500 10 100 10

- calculate pi example

    ./bin/spark-submit
         --master spark://master:7077
         examples/src/main/python/pi.py

# Start to write your own genetic algorithm with spark!
