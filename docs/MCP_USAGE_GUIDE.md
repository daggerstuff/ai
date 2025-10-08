# MCP Server Usage Guide for Amazon Q Developer

## 🚀 **Quick Reference for MCP Servers**

This guide shows how to effectively use the configured MCP servers with Amazon Q Developer for the Pixelated AI project.

---

## 📁 **Core Development Servers**

### **filesystem** (Priority 1)
**Purpose**: Access and manipulate project files
```
Available tools:
- filesystem___read_file
- filesystem___write_file  
- filesystem___create_directory
- filesystem___list_directory
- filesystem___delete_file
- filesystem___move_file
- filesystem___get_file_info
```

**Example Usage**:
- "Read the contents of ai/pixel/models/pixel_base_model.py"
- "List all files in the monitoring directory"
- "Create a new directory for test results"

### **git** (Priority 2)  
**Purpose**: Git repository operations
```
Available tools:
- git___status
- git___log
- git___diff
- git___add
- git___commit
- git___branch
- git___checkout
- git___push
- git___pull
```

**Example Usage**:
- "Show the current git status"
- "Create a new branch for the quality dashboard feature"
- "Commit the changes with message 'Add quality analytics dashboard'"

### **OpenMemory** (Priority 3)
**Purpose**: Context and memory management
```
Available tools:
- OpenMemory___store_memory
- OpenMemory___retrieve_memory
- OpenMemory___search_memories
- OpenMemory___update_memory
- OpenMemory___delete_memory
```

**Example Usage**:
- "Store information about the completed quality dashboard task"
- "Retrieve memories about the Pixel AI architecture"
- "Search for previous discussions about database schema"

---

## 🧠 **Enhanced Capabilities**

### **context7** (Priority 4)
**Purpose**: Advanced context management
```
Available tools:
- context7___store_context
- context7___retrieve_context
- context7___analyze_context
- context7___summarize_context
```

**Example Usage**:
- "Store the current conversation context for later reference"
- "Analyze the context of our quality analytics discussion"

### **sequential-thinking** (Priority 5)
**Purpose**: Step-by-step reasoning and analysis
```
Available tools:
- sequential-thinking___analyze_problem
- sequential-thinking___break_down_task
- sequential-thinking___generate_steps
- sequential-thinking___evaluate_solution
```

**Example Usage**:
- "Break down the task of implementing quality trend analysis"
- "Analyze the problem with the broken dashboard components"

### **time** (Priority 6)
**Purpose**: Date and time utilities
```
Available tools:
- time___get_current_time
- time___format_time
- time___calculate_duration
- time___schedule_reminder
```

**Example Usage**:
- "What's the current timestamp for logging?"
- "Calculate how long the quality dashboard implementation took"

---

## 🔧 **Automation & Testing**

### **playwright** (Priority 7)
**Purpose**: Browser automation and testing
```
Available tools:
- playwright___launch_browser
- playwright___navigate_to_page
- playwright___click_element
- playwright___fill_form
- playwright___take_screenshot
- playwright___run_test
```

**Example Usage**:
- "Test the quality dashboard by navigating to localhost:8501"
- "Take a screenshot of the dashboard interface"
- "Automate testing of the Streamlit interface"

### **browserbase** (Priority 8)
**Purpose**: Cloud browser automation
```
Available tools:
- browserbase___create_session
- browserbase___navigate
- browserbase___interact
- browserbase___capture_data
```

**Example Usage**:
- "Use cloud browser to test the dashboard on different devices"
- "Capture data from external documentation sites"

### **perplexity-search** (Priority 9)
**Purpose**: AI-powered web search
```
Available tools:
- perplexity-search___search
- perplexity-search___ask_question
- perplexity-search___get_sources
```

**Example Usage**:
- "Search for best practices in Streamlit dashboard design"
- "Find information about quality metrics in AI systems"

---

## 📚 **Documentation Access**

### **astro-docs** (Priority 10)
**Purpose**: Astro framework documentation
```
Available tools:
- astro-docs___search_docs
- astro-docs___get_guide
- astro-docs___find_examples
```

**Example Usage**:
- "Find Astro documentation about component architecture"
- "Get examples of Astro integration patterns"

### **huggingface** (Priority 11)
**Purpose**: ML models and datasets access
```
Available tools:
- huggingface___search_models
- huggingface___get_model_info
- huggingface___search_datasets
- huggingface___download_model
```

**Example Usage**:
- "Search for emotion recognition models on Hugging Face"
- "Find datasets for conversation quality assessment"

### **microsoft-docs** (Priority 12)
**Purpose**: Microsoft Learn documentation
```
Available tools:
- microsoft-docs___search_docs
- microsoft-docs___get_article
- microsoft-docs___find_tutorials
```

**Example Usage**:
- "Find Microsoft documentation about Azure deployment"
- "Get tutorials on enterprise application architecture"

---

## 💡 **Best Practices for Using MCP Servers**

### **Efficient Server Usage**
1. **Use Priority Order**: Start with filesystem and git for basic operations
2. **Combine Servers**: Use multiple servers together for complex tasks
3. **Cache Results**: Store frequently accessed information in OpenMemory
4. **Validate Operations**: Use git to track changes made via filesystem

### **Common Workflows**

#### **Development Workflow**
```
1. filesystem___read_file - Read current code
2. sequential-thinking___analyze_problem - Analyze requirements
3. filesystem___write_file - Implement changes
4. git___add + git___commit - Save changes
5. OpenMemory___store_memory - Document decisions
```

#### **Research Workflow**
```
1. perplexity-search___search - Find external information
2. huggingface___search_models - Find relevant models
3. microsoft-docs___search_docs - Check documentation
4. OpenMemory___store_memory - Save research findings
5. filesystem___write_file - Document insights
```

#### **Testing Workflow**
```
1. playwright___launch_browser - Start testing environment
2. playwright___navigate_to_page - Access application
3. playwright___take_screenshot - Capture results
4. filesystem___write_file - Save test reports
5. git___commit - Version control test results
```

---

## 🔍 **Troubleshooting MCP Servers**

### **Common Issues**
1. **Server Not Responding**: Check if command is available in PATH
2. **Permission Errors**: Ensure proper file/directory permissions
3. **API Key Issues**: Verify environment variables are set
4. **Network Timeouts**: Check internet connection for HTTP servers

### **Validation Commands**
```bash
# Validate MCP configuration
python .amazonq/validate_mcp.py

# Check specific server availability
which npx  # For npm-based servers
which uvx  # For Python-based servers
```

### **Debug Mode**
Enable verbose logging by setting log level to "debug" in the MCP configuration defaults section.

---

## 📊 **Server Status Monitoring**

### **Health Check Commands**
- **filesystem**: `filesystem___list_directory /home/vivi/pixelated`
- **git**: `git___status`
- **time**: `time___get_current_time`
- **OpenMemory**: `OpenMemory___retrieve_memory test`

### **Performance Tips**
1. **Use Caching**: Store frequently accessed data in OpenMemory
2. **Batch Operations**: Combine multiple file operations when possible
3. **Async Operations**: Use time server for scheduling non-blocking tasks
4. **Resource Cleanup**: Close browser sessions when done with playwright

---

## ✅ **Ready for Production Use**

The MCP configuration is optimized and validated for Amazon Q Developer. All servers are properly configured with:

- ✅ **Correct protocol implementation**
- ✅ **Proper error handling**
- ✅ **Security best practices**
- ✅ **Comprehensive tool coverage**
- ✅ **Production-ready defaults**

**Start using the MCP servers immediately with Amazon Q Developer for enhanced development capabilities on the Pixelated AI project!**
