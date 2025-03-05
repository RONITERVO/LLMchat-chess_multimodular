class TaskManager:
    def __init__(self):
        self.tasks = []
        self.groups = {}  # Format: {group_name: {'daily_enabled': bool, 'tasks': [task_descs]}}
        self.load_tasks()
    
    def load_tasks(self):
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r') as f:
                data = json.load(f)
                self.tasks = data.get('tasks', [])
                self.groups = data.get('groups', {})
    
    def save_tasks(self):
        data = {
            'tasks': self.tasks,
            'groups': self.groups
        }
        with open(TASKS_FILE, 'w') as f:
            json.dump(data, f, indent=4, default=str)
    
    # Modified add_task to support groups
    def add_task(self, description, due_date, priority="Medium", group=None):
        self.tasks.append({
            "description": description,
            "due_date": due_date.isoformat(),
            "added": datetime.now().isoformat(),
            "priority": priority,
            "completed": False,
            "group": group
        })
        self.save_tasks()
    
    # Method to create/update group
    def add_task_group(self, group_name, task_descriptions, daily_enabled=False):
        self.groups[group_name] = {
            'daily_enabled': daily_enabled,
            'tasks': task_descriptions
        }
        self.save_tasks()
    
    def get_due_tasks(self):
        due = []
        today = datetime.now().date()
        
        # Check existing tasks
        for task in self.tasks:
            if not task['completed'] and datetime.now() > dateutil.parser.parse(task['due_date']):
                due.append(task)
        
        # Process groups for daily activation
        for group_name, config in self.groups.items():
            if config['daily_enabled']:
                # Process each task in the group
                for task_desc in config['tasks']:
                    if not any(t['description'] == task_desc and 
                              t['group'] == group_name and
                              dateutil.parser.parse(t['due_date']).date() == today
                              for t in self.tasks):
                        # Add task if not existing for today
                        self.add_task(
                            description=task_desc,
                            due_date=datetime.now(),
                            priority="Medium",
                            group=group_name
                        )
        
        return due

class ChatAudioApp5(tk.Tk):
    # ... existing code ...
    
    def create_tasks_tab(self):
        # Previous task creation code
        # ... (existing code)
        
        # Add Group Management Section
        group_frame = ttk.LabelFrame(frame, text="Task Groups")
        group_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Group Name
        ttk.Label(group_frame, text="Group Name:").grid(row=0, column=0)
        self.group_name_entry = ttk.Entry(group_frame)
        self.group_name_entry.grid(row=0, column=1)
        
        # Daily Enabled Checkbox
        self.daily_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(group_frame, text="Daily Enabled", variable=self.daily_enabled_var).grid(row=0, column=2)
        
        # Add Group Button
        ttk.Button(group_frame, text="Create/Update Group", 
                  command=self.create_task_group).grid(row=0, column=3)

    def create_task_group(self):
        group_name = self.group_name_entry.get()
        selected_tasks = [self.task_list.item(i)['values'][0] 
                         for i in self.task_list.selection()]
        
        if group_name and selected_tasks:
            self.task_manager.add_task_group(
                group_name,
                selected_tasks,
                self.daily_enabled_var.get()
            )
            messagebox.showinfo("Success", f"Group '{group_name}' updated!")
