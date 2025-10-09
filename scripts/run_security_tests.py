#!/usr/bin/env python3
"""
Security Test Execution Script
Tasks 101 & 102: API Authentication System + Security Test Coverage

This script runs comprehensive security testing for the authentication system
and generates detailed reports for production readiness validation.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_authentication import AuthenticationSystem, AuthenticationTester, UserRole, PermissionLevel
from security_test_framework import SecurityTestFramework

class ComprehensiveSecurityValidator:
    """
    Comprehensive security validation for Tasks 101 & 102
    """
    
    def __init__(self):
        self.results = {}
        self.overall_score = 0.0
        self.production_ready = False
        
    async def run_all_security_tests(self) -> dict:
        """Run all security tests and generate comprehensive report"""
        print("🚀 Starting Comprehensive Security Validation")
        print("=" * 60)
        
        # Task 101: Authentication System Testing
        print("\n📋 TASK 101: API Authentication System Testing")
        auth_results = await self.test_authentication_system()
        
        # Task 102: Security Test Coverage Framework
        print("\n📋 TASK 102: Security Test Coverage Framework")
        security_results = await self.test_security_framework()
        
        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report(auth_results, security_results)
        
        # Save results
        self.save_results(comprehensive_report)
        
        # Print summary
        self.print_summary(comprehensive_report)
        
        return comprehensive_report
    
    async def test_authentication_system(self) -> dict:
        """Test Task 101: API Authentication System"""
        print("  🔐 Initializing Authentication System...")
        
        # Initialize authentication system
        auth_system = AuthenticationSystem(secret_key="test-secret-key-for-validation")
        
        # Create test users and API keys
        admin_user = auth_system.create_user("admin", "admin@test.com", "SecurePassword123!", UserRole.ADMIN)
        regular_user = auth_system.create_user("user", "user@test.com", "UserPassword123!", UserRole.USER)
        readonly_user = auth_system.create_user("readonly", "readonly@test.com", "ReadPassword123!", UserRole.READONLY)
        
        # Create test API keys
        admin_api_key, _ = auth_system.create_api_key(
            "admin_key", [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.DELETE, PermissionLevel.ADMIN]
        )
        user_api_key, _ = auth_system.create_api_key(
            "user_key", [PermissionLevel.READ, PermissionLevel.WRITE]
        )
        readonly_api_key, _ = auth_system.create_api_key(
            "readonly_key", [PermissionLevel.READ]
        )
        
        print("  ✅ Test users and API keys created")
        
        # Run authentication security tests
        print("  🧪 Running Authentication Security Tests...")
        tester = AuthenticationTester(auth_system)
        test_results = tester.run_security_tests()
        
        # Calculate authentication score
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        auth_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"  📊 Authentication Tests: {passed_tests}/{total_tests} passed ({auth_score:.1f}%)")
        
        # Additional validation tests
        validation_results = self.run_authentication_validation_tests(auth_system)
        
        return {
            "task": "Task 101: API Authentication System",
            "score": auth_score,
            "test_results": test_results,
            "validation_results": validation_results,
            "users_created": 3,
            "api_keys_created": 3,
            "production_ready": auth_score >= 90 and all(test_results.values())
        }
    
    def run_authentication_validation_tests(self, auth_system: AuthenticationSystem) -> dict:
        """Run additional authentication validation tests"""
        validation_tests = {}
        
        # Test 1: Password hashing strength
        try:
            password = "TestPassword123!"
            hash1 = auth_system.hash_password(password)
            hash2 = auth_system.hash_password(password)
            
            # Hashes should be different (salted)
            validation_tests["password_salt_unique"] = hash1 != hash2
            
            # Hash should verify correctly
            validation_tests["password_verification"] = auth_system.verify_password(password, hash1)
            
            # Wrong password should not verify
            validation_tests["wrong_password_rejection"] = not auth_system.verify_password("WrongPassword", hash1)
            
        except Exception as e:
            validation_tests["password_hashing_error"] = str(e)
        
        # Test 2: JWT token structure
        try:
            user = list(auth_system.users.values())[0]  # Get first user
            token = auth_system.generate_jwt_token(user)
            
            # Token should have 3 parts (header.payload.signature)
            token_parts = token.split('.')
            validation_tests["jwt_structure_valid"] = len(token_parts) == 3
            
            # Token should verify
            payload = auth_system.verify_jwt_token(token)
            validation_tests["jwt_verification"] = payload is not None
            
            # Payload should contain required fields
            if payload:
                required_fields = ['user_id', 'username', 'role', 'exp', 'iat']
                validation_tests["jwt_payload_complete"] = all(field in payload for field in required_fields)
            
        except Exception as e:
            validation_tests["jwt_token_error"] = str(e)
        
        # Test 3: Role-based access control
        try:
            # Admin should have all permissions
            admin_permissions = all(
                auth_system.check_permission(UserRole.ADMIN, perm) 
                for perm in PermissionLevel
            )
            validation_tests["admin_full_permissions"] = admin_permissions
            
            # Readonly should only have read permission
            readonly_limited = (
                auth_system.check_permission(UserRole.READONLY, PermissionLevel.READ) and
                not auth_system.check_permission(UserRole.READONLY, PermissionLevel.WRITE) and
                not auth_system.check_permission(UserRole.READONLY, PermissionLevel.DELETE)
            )
            validation_tests["readonly_limited_permissions"] = readonly_limited
            
        except Exception as e:
            validation_tests["rbac_error"] = str(e)
        
        # Test 4: API key functionality
        try:
            # Create and authenticate API key
            api_key, api_key_obj = auth_system.create_api_key("test_validation", [PermissionLevel.READ])
            
            # API key should authenticate
            auth_result = auth_system.authenticate_api_key(api_key)
            validation_tests["api_key_authentication"] = auth_result is not None
            
            # API key should have correct permissions
            if auth_result:
                validation_tests["api_key_permissions"] = PermissionLevel.READ in auth_result.permissions
            
            # Invalid API key should not authenticate
            invalid_result = auth_system.authenticate_api_key("invalid_key_12345")
            validation_tests["invalid_api_key_rejection"] = invalid_result is None
            
        except Exception as e:
            validation_tests["api_key_error"] = str(e)
        
        return validation_tests
    
    async def test_security_framework(self) -> dict:
        """Test Task 102: Security Test Coverage Framework"""
        print("  🛡️  Initializing Security Test Framework...")
        
        # Initialize security test framework
        security_tester = SecurityTestFramework("http://localhost:8000")
        
        print("  🧪 Running Security Test Framework...")
        
        # Run individual security test categories
        test_categories = [
            ("Authentication Security", security_tester.test_authentication_security),
            ("Authorization Security", security_tester.test_authorization_security),
            ("Input Validation Security", security_tester.test_input_validation_security),
            ("Session Management Security", security_tester.test_session_management_security),
        ]
        
        category_results = {}
        total_vulnerabilities = 0
        
        for category_name, test_method in test_categories:
            try:
                print(f"    🔍 Testing {category_name}...")
                result = await test_method()
                category_results[category_name] = {
                    "passed": result.passed,
                    "vulnerabilities_found": len(result.vulnerabilities),
                    "execution_time": result.execution_time
                }
                total_vulnerabilities += len(result.vulnerabilities)
                security_tester.vulnerabilities.extend(result.vulnerabilities)
                
            except Exception as e:
                print(f"    ❌ {category_name} failed: {e}")
                category_results[category_name] = {
                    "passed": False,
                    "error": str(e),
                    "vulnerabilities_found": 0,
                    "execution_time": 0.0
                }
        
        # Calculate security framework score
        passed_categories = sum(1 for result in category_results.values() if result.get("passed", False))
        total_categories = len(category_results)
        framework_score = (passed_categories / total_categories) * 100 if total_categories > 0 else 0
        
        # Adjust score based on vulnerabilities found
        if total_vulnerabilities > 0:
            vulnerability_penalty = min(total_vulnerabilities * 5, 30)  # Max 30 point penalty
            framework_score = max(0, framework_score - vulnerability_penalty)
        
        print(f"  📊 Security Framework Tests: {passed_categories}/{total_categories} passed ({framework_score:.1f}%)")
        print(f"  🚨 Total Vulnerabilities Found: {total_vulnerabilities}")
        
        return {
            "task": "Task 102: Security Test Coverage Framework",
            "score": framework_score,
            "category_results": category_results,
            "total_vulnerabilities": total_vulnerabilities,
            "vulnerabilities": [
                {
                    "id": vuln.id,
                    "title": vuln.title,
                    "severity": vuln.severity.value,
                    "category": vuln.category.value
                }
                for vuln in security_tester.vulnerabilities
            ],
            "production_ready": framework_score >= 90 and total_vulnerabilities == 0
        }
    
    def generate_comprehensive_report(self, auth_results: dict, security_results: dict) -> dict:
        """Generate comprehensive security validation report"""
        
        # Calculate overall score (weighted average)
        auth_weight = 0.6  # Authentication is critical
        security_weight = 0.4  # Security testing is important
        
        overall_score = (
            auth_results["score"] * auth_weight +
            security_results["score"] * security_weight
        )
        
        # Determine production readiness
        production_ready = (
            auth_results["production_ready"] and
            security_results["production_ready"] and
            overall_score >= 90
        )
        
        # Generate status
        if production_ready:
            status = "✅ PRODUCTION READY"
        elif overall_score >= 80:
            status = "⚠️  NEEDS MINOR FIXES"
        elif overall_score >= 60:
            status = "🔧 NEEDS MAJOR FIXES"
        else:
            status = "❌ NOT PRODUCTION READY"
        
        return {
            "validation_summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_score": round(overall_score, 1),
                "status": status,
                "production_ready": production_ready
            },
            "task_results": {
                "task_101_authentication": auth_results,
                "task_102_security_framework": security_results
            },
            "security_metrics": {
                "authentication_score": auth_results["score"],
                "security_framework_score": security_results["score"],
                "total_vulnerabilities": security_results["total_vulnerabilities"],
                "critical_vulnerabilities": len([
                    v for v in security_results["vulnerabilities"] 
                    if v["severity"] == "critical"
                ])
            },
            "recommendations": self.generate_recommendations(overall_score, security_results["total_vulnerabilities"]),
            "next_steps": self.generate_next_steps(production_ready, overall_score)
        }
    
    def generate_recommendations(self, overall_score: float, total_vulnerabilities: int) -> list:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if overall_score < 90:
            recommendations.append("Address failing security tests before production deployment")
        
        if total_vulnerabilities > 0:
            recommendations.append(f"Fix {total_vulnerabilities} security vulnerabilities identified")
        
        if overall_score < 80:
            recommendations.append("Conduct additional security review and testing")
        
        recommendations.extend([
            "Implement continuous security monitoring",
            "Set up automated security testing in CI/CD pipeline",
            "Establish incident response procedures",
            "Schedule regular security audits"
        ])
        
        return recommendations
    
    def generate_next_steps(self, production_ready: bool, overall_score: float) -> list:
        """Generate next steps based on results"""
        if production_ready:
            return [
                "✅ Tasks 101 & 102 completed successfully",
                "🚀 Ready to proceed with Task 103: Safety Validation Certification",
                "📋 Continue with Phase 1 critical security tasks",
                "🔄 Set up continuous security monitoring"
            ]
        else:
            next_steps = [
                "🔧 Fix identified security issues",
                "🧪 Re-run security validation tests",
                "📋 Review authentication system implementation"
            ]
            
            if overall_score < 80:
                next_steps.append("🔍 Conduct comprehensive security code review")
            
            next_steps.append("🔄 Repeat validation until production ready")
            
            return next_steps
    
    def save_results(self, report: dict):
        """Save test results to files"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report_file = f"security_validation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  💾 Comprehensive report saved: {report_file}")
        
        # Save summary report
        summary = {
            "timestamp": report["validation_summary"]["timestamp"],
            "overall_score": report["validation_summary"]["overall_score"],
            "status": report["validation_summary"]["status"],
            "production_ready": report["validation_summary"]["production_ready"],
            "task_101_score": report["task_results"]["task_101_authentication"]["score"],
            "task_102_score": report["task_results"]["task_102_security_framework"]["score"],
            "total_vulnerabilities": report["security_metrics"]["total_vulnerabilities"]
        }
        
        summary_file = f"security_validation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  💾 Summary report saved: {summary_file}")
    
    def print_summary(self, report: dict):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🛡️  COMPREHENSIVE SECURITY VALIDATION REPORT")
        print("=" * 60)
        
        summary = report["validation_summary"]
        print(f"📊 Overall Score: {summary['overall_score']}/100")
        print(f"🎯 Status: {summary['status']}")
        print(f"🚀 Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
        
        print(f"\n📋 TASK BREAKDOWN:")
        task_101 = report["task_results"]["task_101_authentication"]
        task_102 = report["task_results"]["task_102_security_framework"]
        
        print(f"  Task 101 (Authentication): {task_101['score']:.1f}/100 {'✅' if task_101['production_ready'] else '❌'}")
        print(f"  Task 102 (Security Framework): {task_102['score']:.1f}/100 {'✅' if task_102['production_ready'] else '❌'}")
        
        print(f"\n🚨 SECURITY METRICS:")
        metrics = report["security_metrics"]
        print(f"  Total Vulnerabilities: {metrics['total_vulnerabilities']}")
        print(f"  Critical Vulnerabilities: {metrics['critical_vulnerabilities']}")
        
        if report["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        if report["next_steps"]:
            print(f"\n🎯 NEXT STEPS:")
            for step in report["next_steps"]:
                print(f"  {step}")
        
        print("\n" + "=" * 60)

async def main():
    """Main execution function"""
    validator = ComprehensiveSecurityValidator()
    
    try:
        # Run comprehensive security validation
        report = await validator.run_all_security_tests()
        
        # Return appropriate exit code
        if report["validation_summary"]["production_ready"]:
            print("\n🎉 SUCCESS: Tasks 101 & 102 completed successfully!")
            print("✅ Ready for production deployment")
            sys.exit(0)
        else:
            print("\n⚠️  WARNING: Security validation incomplete")
            print("🔧 Additional work required before production")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR: Security validation failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    print("🚀 Starting Tasks 101 & 102: Security Validation")
    print("📋 API Authentication System + Security Test Coverage Framework")
    print("⏰ Starting at:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    
    asyncio.run(main())
