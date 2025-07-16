-- MySQL dump 10.13  Distrib 8.0.41, for Win64 (x86_64)
--
-- Host: localhost    Database: medisense_db
-- ------------------------------------------------------
-- Server version	8.0.41

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `accounts_activitylog`
--

DROP TABLE IF EXISTS `accounts_activitylog`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `accounts_activitylog` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `action` varchar(255) NOT NULL,
  `timestamp` datetime(6) NOT NULL,
  `description` longtext,
  `user_id` int NOT NULL,
  PRIMARY KEY (`id`),
  KEY `accounts_activitylog_user_id_2d7b43c7_fk_auth_user_id` (`user_id`),
  CONSTRAINT `accounts_activitylog_user_id_2d7b43c7_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=57 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `accounts_activitylog`
--

LOCK TABLES `accounts_activitylog` WRITE;
/*!40000 ALTER TABLE `accounts_activitylog` DISABLE KEYS */;
INSERT INTO `accounts_activitylog` VALUES (1,'login','2025-07-12 21:17:16.570323',NULL,1),(2,'login','2025-07-12 21:20:18.088689',NULL,1),(3,'updated','2025-07-12 21:20:26.629618','Medicine: fds (updated)',1),(4,'deleted','2025-07-12 21:20:26.631616','Medicine: fds (deleted)',1),(5,'login','2025-07-12 21:23:41.607802',NULL,3),(6,'added','2025-07-12 21:24:22.984740','Medicine: here (added)',3),(7,'deleted','2025-07-12 21:26:57.637989','Medicine: here (deleted)',3),(8,'login','2025-07-12 21:32:09.540545',NULL,1),(9,'login','2025-07-12 21:54:21.759626','User bryllenyel logged out',1),(10,'login','2025-07-13 03:34:35.064879','User bryllenyel logged out',1),(11,'logout','2025-07-13 03:34:59.684391','User bryllenyel logged out',1),(12,'added','2025-07-13 09:33:52.022791','Inventory: d (added)',1),(13,'updated','2025-07-13 09:33:58.047128','Inventory: d (updated)',1),(14,'deleted','2025-07-13 09:33:58.049125','Inventory: d (deleted)',1),(15,'added','2025-07-13 10:45:43.503770','Inventory: rte (added)',1),(16,'updated','2025-07-13 10:46:04.440007','Inventory: rte (updated)',1),(17,'deleted','2025-07-13 10:46:04.442880','Inventory: rte (deleted)',1),(18,'added','2025-07-13 10:51:10.452969','Inventory: ad (added)',1),(19,'updated','2025-07-13 10:51:17.306700','Inventory: ad (updated)',1),(20,'deleted','2025-07-13 10:51:17.310512','Inventory: ad (deleted)',1),(21,'added','2025-07-13 10:51:23.512526','Inventory: 213 (added)',1),(22,'updated','2025-07-13 10:56:34.528606','Inventory: 213 (updated)',1),(23,'deleted','2025-07-13 10:56:34.530613','Inventory: 213 (deleted)',1),(24,'added','2025-07-13 10:56:42.729373','Inventory: 12 (added)',1),(25,'updated','2025-07-13 10:56:48.872318','Inventory: 12 (updated)',1),(26,'deleted','2025-07-13 10:56:48.875282','Inventory: 12 (deleted)',1),(27,'added','2025-07-13 10:56:55.677185','Inventory: 13 (added)',1),(28,'updated','2025-07-13 10:59:29.311736','Inventory: 13 (updated)',1),(29,'updated','2025-07-13 11:02:28.971358','Inventory: 13 (updated)',1),(30,'updated','2025-07-13 11:05:10.545646','Inventory: 13 (updated)',1),(31,'deleted','2025-07-13 11:05:10.547644','Inventory: 13 (deleted)',1),(32,'added','2025-07-13 11:05:36.006018','Inventory: 324 (added)',1),(33,'updated','2025-07-13 11:09:01.025257','Inventory: 324 (updated)',1),(34,'login','2025-07-13 13:09:14.186250','User bryllenyel logged out',1),(35,'added','2025-07-13 13:09:31.152231','Inventory: sdf (added)',1),(36,'updated','2025-07-13 13:09:42.700401','Inventory: 324 (updated)',1),(37,'deleted','2025-07-13 13:09:42.703396','Inventory: 324 (deleted)',1),(38,'updated','2025-07-13 13:09:43.241159','Inventory: sdf (updated)',1),(39,'deleted','2025-07-13 13:09:43.243160','Inventory: sdf (deleted)',1),(40,'added','2025-07-13 14:38:51.706022','Inventory: Medicine X (added)',1),(41,'login','2025-07-13 14:44:13.400008','User bryllenyel logged out',1),(42,'login','2025-07-13 16:49:39.520223','User bryllenyel logged out',1),(43,'login','2025-07-13 17:05:32.676287','User hajime logged out',5),(44,'login','2025-07-13 17:06:41.942123','User bryllenyel logged out',1),(45,'logout','2025-07-13 17:18:51.353528','User bryllenyel logged out',1),(46,'login','2025-07-15 09:23:20.075272','User bryllenyel logged out',1),(47,'login','2025-07-15 12:11:18.930785','User bryllenyel logged out',1),(48,'login','2025-07-15 12:49:40.036299','User bryllenyel logged out',1),(49,'updated','2025-07-15 13:44:11.610890','Inventory: Medicine X (updated)',1),(50,'deleted','2025-07-15 13:44:11.613904','Inventory: Medicine X (deleted)',1),(51,'added','2025-07-15 13:44:21.434019','Inventory: 3242 (added)',1),(52,'login','2025-07-15 14:22:15.727841','User bryllenyel logged out',1),(53,'login','2025-07-15 14:59:27.483149','User bryllenyel logged out',1),(54,'login','2025-07-15 15:00:34.058541','User edizen logged out',6),(55,'login','2025-07-15 15:14:35.698613','User bryllenyel logged out',1),(56,'login','2025-07-16 05:04:29.350927','User bryllenyel logged out',1);
/*!40000 ALTER TABLE `accounts_activitylog` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_group`
--

DROP TABLE IF EXISTS `auth_group`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_group` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_group`
--

LOCK TABLES `auth_group` WRITE;
/*!40000 ALTER TABLE `auth_group` DISABLE KEYS */;
INSERT INTO `auth_group` VALUES (1,'Users');
/*!40000 ALTER TABLE `auth_group` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_group_permissions`
--

DROP TABLE IF EXISTS `auth_group_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_group_permissions` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `group_id` int NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`),
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_group_permissions`
--

LOCK TABLES `auth_group_permissions` WRITE;
/*!40000 ALTER TABLE `auth_group_permissions` DISABLE KEYS */;
INSERT INTO `auth_group_permissions` VALUES (10,1,29),(11,1,30),(12,1,31),(1,1,32),(2,1,33),(3,1,34),(4,1,35),(5,1,36),(6,1,41),(7,1,42),(8,1,43),(9,1,44);
/*!40000 ALTER TABLE `auth_group_permissions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_permission`
--

DROP TABLE IF EXISTS `auth_permission`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_permission` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=53 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_permission`
--

LOCK TABLES `auth_permission` WRITE;
/*!40000 ALTER TABLE `auth_permission` DISABLE KEYS */;
INSERT INTO `auth_permission` VALUES (1,'Can add log entry',1,'add_logentry'),(2,'Can change log entry',1,'change_logentry'),(3,'Can delete log entry',1,'delete_logentry'),(4,'Can view log entry',1,'view_logentry'),(5,'Can add permission',2,'add_permission'),(6,'Can change permission',2,'change_permission'),(7,'Can delete permission',2,'delete_permission'),(8,'Can view permission',2,'view_permission'),(9,'Can add group',3,'add_group'),(10,'Can change group',3,'change_group'),(11,'Can delete group',3,'delete_group'),(12,'Can view group',3,'view_group'),(13,'Can add user',4,'add_user'),(14,'Can change user',4,'change_user'),(15,'Can delete user',4,'delete_user'),(16,'Can view user',4,'view_user'),(17,'Can add content type',5,'add_contenttype'),(18,'Can change content type',5,'change_contenttype'),(19,'Can delete content type',5,'delete_contenttype'),(20,'Can view content type',5,'view_contenttype'),(21,'Can add session',6,'add_session'),(22,'Can change session',6,'change_session'),(23,'Can delete session',6,'delete_session'),(24,'Can view session',6,'view_session'),(25,'Can add user',7,'add_user'),(26,'Can change user',7,'change_user'),(27,'Can delete user',7,'delete_user'),(28,'Can view user',7,'view_user'),(29,'Can add patient',8,'add_patient'),(30,'Can change patient',8,'change_patient'),(31,'Can delete patient',8,'delete_patient'),(32,'Can view patient',8,'view_patient'),(33,'Can add symptom log',9,'add_symptomlog'),(34,'Can change symptom log',9,'change_symptomlog'),(35,'Can delete symptom log',9,'delete_symptomlog'),(36,'Can view symptom log',9,'view_symptomlog'),(37,'Can add prediction',10,'add_prediction'),(38,'Can change prediction',10,'change_prediction'),(39,'Can delete prediction',10,'delete_prediction'),(40,'Can view prediction',10,'view_prediction'),(41,'Can add inventory item',11,'add_inventoryitem'),(42,'Can change inventory item',11,'change_inventoryitem'),(43,'Can delete inventory item',11,'delete_inventoryitem'),(44,'Can view inventory item',11,'view_inventoryitem'),(45,'Can add captcha store',12,'add_captchastore'),(46,'Can change captcha store',12,'change_captchastore'),(47,'Can delete captcha store',12,'delete_captchastore'),(48,'Can view captcha store',12,'view_captchastore'),(49,'Can add activity log',13,'add_activitylog'),(50,'Can change activity log',13,'change_activitylog'),(51,'Can delete activity log',13,'delete_activitylog'),(52,'Can view activity log',13,'view_activitylog');
/*!40000 ALTER TABLE `auth_permission` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_user`
--

DROP TABLE IF EXISTS `auth_user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_user` (
  `id` int NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(150) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_user`
--

LOCK TABLES `auth_user` WRITE;
/*!40000 ALTER TABLE `auth_user` DISABLE KEYS */;
INSERT INTO `auth_user` VALUES (1,'pbkdf2_sha256$1000000$L8F8dCMm15aMh8GIAtLyuw$apoJAIwpfmr9CPdNL4bQL/OsVeC/Tud5QPcs7MK5+kU=','2025-07-16 05:04:29.342883',1,'bryllenyel','Brylle Nyel','Mamuad','bryllenyelm@gmail.com',1,1,'2025-07-05 19:53:36.000000'),(2,'pbkdf2_sha256$1000000$1SxB4N5TTvv7XA6vWEyFCb$QcxggoUMcD5b8CKA3RBpPDy9r6/BPewzTFYtVlx+Spk=','2025-07-12 19:36:41.709324',0,'markangelo','Mark Angelo','Daquioag','markangelod@gmail.com',1,1,'2025-07-12 16:45:03.000000'),(3,'pbkdf2_sha256$1000000$NbCWLMiNm5zWmt0SWWAZsc$Pld3UruCG1op0klDBIiPp6eCM7OzYgYD30Yf+SdoDIU=','2025-07-12 21:23:41.607802',0,'james','','','jamesbond@gmail.com',0,1,'2025-07-12 20:22:37.000000'),(4,'pbkdf2_sha256$1000000$C7Voly9H08W8UN71YIbtWw$SBO8OLut+VC5g+CP1uwsLiTaz5FlVsPvGXC8rUYE0Yg=','2025-07-12 20:33:31.713646',0,'joel','','','joelisdead2@gmail.com',0,1,'2025-07-12 20:30:43.000000'),(5,'pbkdf2_sha256$1000000$tOQxbxlPcPTYBJxzSf0sQT$v+bVJklWy1KiA/vvNvzsbw0DZHpD7Qjhh1C0bSBtPLg=','2025-07-13 17:05:32.675290',0,'hajime','','','bryllenyelmamuad05@gmail.com',0,1,'2025-07-13 16:49:27.000000'),(6,'pbkdf2_sha256$1000000$c0zQzjosjN6XBVlGUDsb8M$THsF1ArCyGSQ8A6PuUhwkHg9pQfSmfP6YXltc1vR94c=','2025-07-15 15:00:34.057367',0,'edizen','','','edizen@gmail.com',0,1,'2025-07-15 14:59:07.000000');
/*!40000 ALTER TABLE `auth_user` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_user_groups`
--

DROP TABLE IF EXISTS `auth_user_groups`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_user_groups` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `group_id` int NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_groups_user_id_group_id_94350c0c_uniq` (`user_id`,`group_id`),
  KEY `auth_user_groups_group_id_97559544_fk_auth_group_id` (`group_id`),
  CONSTRAINT `auth_user_groups_group_id_97559544_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`),
  CONSTRAINT `auth_user_groups_user_id_6a12ed8b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_user_groups`
--

LOCK TABLES `auth_user_groups` WRITE;
/*!40000 ALTER TABLE `auth_user_groups` DISABLE KEYS */;
INSERT INTO `auth_user_groups` VALUES (1,4,1);
/*!40000 ALTER TABLE `auth_user_groups` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `auth_user_user_permissions`
--

DROP TABLE IF EXISTS `auth_user_user_permissions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `auth_user_user_permissions` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` (`user_id`,`permission_id`),
  KEY `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` (`permission_id`),
  CONSTRAINT `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  CONSTRAINT `auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `auth_user_user_permissions`
--

LOCK TABLES `auth_user_user_permissions` WRITE;
/*!40000 ALTER TABLE `auth_user_user_permissions` DISABLE KEYS */;
/*!40000 ALTER TABLE `auth_user_user_permissions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `captcha_captchastore`
--

DROP TABLE IF EXISTS `captcha_captchastore`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `captcha_captchastore` (
  `id` int NOT NULL AUTO_INCREMENT,
  `challenge` varchar(32) NOT NULL,
  `response` varchar(32) NOT NULL,
  `hashkey` varchar(40) NOT NULL,
  `expiration` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `hashkey` (`hashkey`)
) ENGINE=InnoDB AUTO_INCREMENT=361 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `captcha_captchastore`
--

LOCK TABLES `captcha_captchastore` WRITE;
/*!40000 ALTER TABLE `captcha_captchastore` DISABLE KEYS */;
INSERT INTO `captcha_captchastore` VALUES (358,'ZGUU','zguu','831224fa6e96fe9db1953f27a140be71e211895b','2025-07-16 05:09:04.016367'),(359,'DJWIYMTRGMVAXGBFXDPV','djwiymtrgmvaxgbfxdpv','16125b3ad2d99cf13c592a7a85075496c3136282','2025-07-16 05:09:13.559355');
/*!40000 ALTER TABLE `captcha_captchastore` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_admin_log`
--

DROP TABLE IF EXISTS `django_admin_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_admin_log` (
  `id` int NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint unsigned NOT NULL,
  `change_message` longtext NOT NULL,
  `content_type_id` int DEFAULT NULL,
  `user_id` int NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb_fk_django_co` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6_fk_auth_user_id` (`user_id`),
  CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`),
  CONSTRAINT `django_admin_log_user_id_c564eba6_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `django_admin_log_chk_1` CHECK ((`action_flag` >= 0))
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_admin_log`
--

LOCK TABLES `django_admin_log` WRITE;
/*!40000 ALTER TABLE `django_admin_log` DISABLE KEYS */;
INSERT INTO `django_admin_log` VALUES (1,'2025-07-05 20:59:38.098582','1','Mark Angelo',1,'[{\"added\": {}}]',7,1),(2,'2025-07-05 21:00:16.782687','2','Brylle',1,'[{\"added\": {}}]',7,1),(3,'2025-07-05 21:00:34.081843','1','Mark Angelo (edited)',2,'[{\"changed\": {\"fields\": [\"Username\"]}}]',7,1),(4,'2025-07-05 21:00:51.349262','1','Mark Angelo (edited)',3,'',7,1),(5,'2025-07-05 21:00:54.558776','2','Brylle',3,'',7,1),(6,'2025-07-05 21:01:02.539263','3','sdf',1,'[{\"added\": {}}]',7,1),(7,'2025-07-05 21:01:11.427461','3','sdf',3,'',7,1),(8,'2025-07-12 16:11:29.489389','4','bryllenyel',1,'[{\"added\": {}}]',7,1),(9,'2025-07-12 16:42:12.723294','1','bryllenyel',2,'[{\"changed\": {\"fields\": [\"First name\", \"Last name\", \"Last login\"]}}]',4,1),(10,'2025-07-12 16:45:04.280366','2','markangelo',1,'[{\"added\": {}}]',4,1),(11,'2025-07-12 16:45:48.670825','2','markangelo',2,'[{\"changed\": {\"fields\": [\"First name\", \"Last name\", \"Email address\", \"Staff status\"]}}]',4,1),(12,'2025-07-12 20:24:20.631143','3','james',2,'[{\"changed\": {\"fields\": [\"Active\"]}}]',4,1),(13,'2025-07-12 20:32:09.932421','1','Users',1,'[{\"added\": {}}]',3,1),(14,'2025-07-12 20:32:24.774930','4','joel',2,'[{\"changed\": {\"fields\": [\"Active\", \"Groups\"]}}]',4,1),(15,'2025-07-12 21:26:57.635990','21','here',3,'',11,1),(16,'2025-07-13 16:49:51.777682','5','hajime',2,'[{\"changed\": {\"fields\": [\"Active\"]}}]',4,1),(17,'2025-07-15 14:59:38.218903','6','edizen',2,'[{\"changed\": {\"fields\": [\"Active\"]}}]',4,1);
/*!40000 ALTER TABLE `django_admin_log` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_content_type`
--

DROP TABLE IF EXISTS `django_content_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_content_type` (
  `id` int NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_content_type`
--

LOCK TABLES `django_content_type` WRITE;
/*!40000 ALTER TABLE `django_content_type` DISABLE KEYS */;
INSERT INTO `django_content_type` VALUES (13,'accounts','activitylog'),(7,'accounts','user'),(1,'admin','logentry'),(3,'auth','group'),(2,'auth','permission'),(4,'auth','user'),(12,'captcha','captchastore'),(5,'contenttypes','contenttype'),(11,'inventory','inventoryitem'),(8,'patients','patient'),(9,'patients','symptomlog'),(10,'predictor','prediction'),(6,'sessions','session');
/*!40000 ALTER TABLE `django_content_type` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_migrations`
--

DROP TABLE IF EXISTS `django_migrations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_migrations` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=29 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_migrations`
--

LOCK TABLES `django_migrations` WRITE;
/*!40000 ALTER TABLE `django_migrations` DISABLE KEYS */;
INSERT INTO `django_migrations` VALUES (1,'accounts','0001_initial','2025-06-29 14:15:01.218445'),(2,'contenttypes','0001_initial','2025-06-29 14:15:01.265450'),(3,'auth','0001_initial','2025-06-29 14:15:01.683625'),(4,'admin','0001_initial','2025-06-29 14:15:01.789149'),(5,'admin','0002_logentry_remove_auto_add','2025-06-29 14:15:01.796145'),(6,'admin','0003_logentry_add_action_flag_choices','2025-06-29 14:15:01.815689'),(7,'contenttypes','0002_remove_content_type_name','2025-06-29 14:15:01.943091'),(8,'auth','0002_alter_permission_name_max_length','2025-06-29 14:15:01.995096'),(9,'auth','0003_alter_user_email_max_length','2025-06-29 14:15:02.019648'),(10,'auth','0004_alter_user_username_opts','2025-06-29 14:15:02.026648'),(11,'auth','0005_alter_user_last_login_null','2025-06-29 14:15:02.072646'),(12,'auth','0006_require_contenttypes_0002','2025-06-29 14:15:02.074644'),(13,'auth','0007_alter_validators_add_error_messages','2025-06-29 14:15:02.080647'),(14,'auth','0008_alter_user_username_max_length','2025-06-29 14:15:02.130185'),(15,'auth','0009_alter_user_last_name_max_length','2025-06-29 14:15:02.187185'),(16,'auth','0010_alter_group_name_max_length','2025-06-29 14:15:02.204195'),(17,'auth','0011_update_proxy_permissions','2025-06-29 14:15:02.211190'),(18,'auth','0012_alter_user_first_name_max_length','2025-06-29 14:15:02.280278'),(19,'sessions','0001_initial','2025-06-29 14:15:02.312310'),(20,'inventory','0001_initial','2025-06-29 14:16:35.577448'),(21,'patients','0001_initial','2025-06-29 14:16:35.645085'),(22,'predictor','0001_initial','2025-06-29 14:16:35.701078'),(23,'inventory','0002_remove_inventoryitem_budget_and_more','2025-07-05 19:19:35.273207'),(24,'accounts','0002_delete_user','2025-07-12 16:27:12.882496'),(25,'captcha','0001_initial','2025-07-12 17:28:07.445016'),(26,'captcha','0002_alter_captchastore_id','2025-07-12 17:28:07.445016'),(27,'accounts','0003_initial','2025-07-12 21:01:44.774231'),(28,'inventory','0003_inventoryitem_last_modified_by_and_more','2025-07-12 21:19:46.494364');
/*!40000 ALTER TABLE `django_migrations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `django_session`
--

DROP TABLE IF EXISTS `django_session`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `django_session`
--

LOCK TABLES `django_session` WRITE;
/*!40000 ALTER TABLE `django_session` DISABLE KEYS */;
INSERT INTO `django_session` VALUES ('98dq9uv3rfgjvjb502gvz3xcumi0z8o9','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1uagoE:82CT5kTvXpolh9fY4iHFxO17Z-4kYalE18bmRbJBTYA','2025-07-26 20:26:58.270157'),('b0bo68ol36bfzpc1g98ppo5gnjl44tbt','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1uahpJ:ae7Z6JWKwtuP8x1IOZY9V2ikZnmhugMq-y45yNL8XjU','2025-07-26 21:32:09.541877'),('ful9dbgj36f2nr7p9zg7amazqcbfrh2f','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1ubeVC:wkeedr6jXPoup53fqDQjXl-6-KslhVpeG8CVAFTUH30','2025-07-29 12:11:18.930785'),('loyt2980ikj6bjzmqma5o6fy8n3dvlqt','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1uawSA:6bMW5MSdK3ko584HQwx5wXJqcYUe4KUTyuLd1--LjDE','2025-07-27 13:09:14.188250'),('ofj0pz61of3tumez1birisv7v521fq21','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1uae1w:AZHO1a7quPlO1IJAe83FpxuArMNgQ3w_ok0Ij0PMDkI','2025-07-26 17:28:56.268347'),('pb98cad0cbnfxomdt1yv9t66vo9n7dft','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1ubuJh:XiTor-ZbmK4j2_3baSHwkL24JazvVFFrcVm1inUZIqQ','2025-07-30 05:04:29.359052'),('uu6960u8odempsbeyx3em5g56qze35yx','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1uahG6:F_aGqlrN5Y5mf47OpAhBxkmboPq30OsJQsH-fDJoBnI','2025-07-26 20:55:46.390287'),('yt02pw4xf9shllloof5authase7ank7g','.eJxVjEEOwiAQRe_C2hAGplBcuvcMZDqAVA0kpV0Z765NutDtf-_9lwi0rSVsPS1hjuIsQJx-t4n4keoO4p3qrUludV3mSe6KPGiX1xbT83K4fweFevnWZLwejCeADArzEJ21FtgjGYfRJsasjSJlHPuRITurlNag84jOJLTi_QG35TZ8:1ubgXv:FAH2QJOI6cLrWAfRE3cIqMmK1xxhObUNCK8mdqzIftE','2025-07-29 14:22:15.727841');
/*!40000 ALTER TABLE `django_session` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `inventory_inventoryitem`
--

DROP TABLE IF EXISTS `inventory_inventoryitem`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `inventory_inventoryitem` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `category` varchar(50) NOT NULL,
  `date_added` date NOT NULL,
  `quantity` int unsigned NOT NULL,
  `status` varchar(20) NOT NULL,
  `unit` varchar(20) NOT NULL,
  `last_modified_by_id` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `inventory_inventoryi_last_modified_by_id_6390e2e1_fk_auth_user` (`last_modified_by_id`),
  CONSTRAINT `inventory_inventoryi_last_modified_by_id_6390e2e1_fk_auth_user` FOREIGN KEY (`last_modified_by_id`) REFERENCES `auth_user` (`id`),
  CONSTRAINT `inventory_inventoryitem_chk_1` CHECK ((`quantity` >= 0))
) ENGINE=InnoDB AUTO_INCREMENT=32 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `inventory_inventoryitem`
--

LOCK TABLES `inventory_inventoryitem` WRITE;
/*!40000 ALTER TABLE `inventory_inventoryitem` DISABLE KEYS */;
INSERT INTO `inventory_inventoryitem` VALUES (31,'3242','Painkiller','2025-07-15',234,'Low Stock','pcs',1);
/*!40000 ALTER TABLE `inventory_inventoryitem` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `patients_patient`
--

DROP TABLE IF EXISTS `patients_patient`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patients_patient` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `age` int NOT NULL,
  `sex` varchar(10) NOT NULL,
  `date_logged` date NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `patients_patient`
--

LOCK TABLES `patients_patient` WRITE;
/*!40000 ALTER TABLE `patients_patient` DISABLE KEYS */;
/*!40000 ALTER TABLE `patients_patient` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `patients_symptomlog`
--

DROP TABLE IF EXISTS `patients_symptomlog`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patients_symptomlog` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `symptom` varchar(255) NOT NULL,
  `notes` longtext NOT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`id`),
  KEY `patients_symptomlog_patient_id_706d120c_fk_patients_patient_id` (`patient_id`),
  CONSTRAINT `patients_symptomlog_patient_id_706d120c_fk_patients_patient_id` FOREIGN KEY (`patient_id`) REFERENCES `patients_patient` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `patients_symptomlog`
--

LOCK TABLES `patients_symptomlog` WRITE;
/*!40000 ALTER TABLE `patients_symptomlog` DISABLE KEYS */;
/*!40000 ALTER TABLE `patients_symptomlog` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `predictor_prediction`
--

DROP TABLE IF EXISTS `predictor_prediction`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `predictor_prediction` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `predicted_medicine` varchar(100) NOT NULL,
  `confidence_score` double NOT NULL,
  `date_predicted` date NOT NULL,
  `patient_id` bigint NOT NULL,
  PRIMARY KEY (`id`),
  KEY `predictor_prediction_patient_id_6d815d87_fk_patients_patient_id` (`patient_id`),
  CONSTRAINT `predictor_prediction_patient_id_6d815d87_fk_patients_patient_id` FOREIGN KEY (`patient_id`) REFERENCES `patients_patient` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `predictor_prediction`
--

LOCK TABLES `predictor_prediction` WRITE;
/*!40000 ALTER TABLE `predictor_prediction` DISABLE KEYS */;
/*!40000 ALTER TABLE `predictor_prediction` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-07-16 13:04:41
