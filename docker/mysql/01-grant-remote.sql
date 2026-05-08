-- MySQL 8 init script: re-grant the app user from any host.
-- MYSQL_USER is created by the official image bound to 'localhost'.
-- This runs at first-boot and widens the grant to '%'.
-- Mounted at /docker-entrypoint-initdb.d/

CREATE USER IF NOT EXISTS 'ragwebui'@'%' IDENTIFIED BY 'ragwebui';
GRANT ALL PRIVILEGES ON `ragwebui`.* TO 'ragwebui'@'%';
FLUSH PRIVILEGES;
