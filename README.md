# Medisense - Medicine Inventory Management System

A Django-based inventory and logging system for managing clinic medicine supply, symptoms log, and basic predictive interface.

## Features

- ✅ Login and role-based access control (Admin, Staff)
- ✅ Medicine Inventory Table (CRUD)
- ✅ Export to Excel
- ✅ Search, Sort, Filter
- ✅ Activity Log (WIP)
- ✅ Backup and Restore (WIP)
- ✅ Symptom Logging (WIP)
- ✅ Prediction UI (Prototype)
- ✅ MySQL database integration

## Setup Instructions

### 1. Clone Repository

``` bash
git clone https://github.com/brynyz/medisense.git
cd medisense
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure `.env`

Create a `.env` file and add your database credentials:

```env
DB_NAME=your_db_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
```

### 5. Apply Migrations and Run Server

```bash
python manage.py migrate
python manage.py runserver
```

## Notes

- Make sure MySQL server is running and the database exists.
- Admin dashboard: `/admin`
- Main access after login: `/`

## To Do

- [ ] Popup modals for Add/Edit forms
- [ ] Prevent duplicate medicine entries
- [ ] In-row editing (optional)
- [ ] Implement full logging and backup features
- [ ] Finalize Prediction prototype UI

---

© 2025 Medisense
```

Let me know if you want to include example screenshots or contributor sections.
