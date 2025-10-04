#!/bin/bash

# DigitalOcean Deployment Script for MediSense
# Run this script to deploy your application

set -e  # Exit on any error

echo "🚀 Starting MediSense deployment to DigitalOcean..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required tools are installed
check_requirements() {
    echo "📋 Checking requirements..."
    
    if ! command -v doctl &> /dev/null; then
        echo -e "${RED}❌ doctl CLI not found. Please install it first:${NC}"
        echo "   curl -sL https://github.com/digitalocean/doctl/releases/download/v1.94.0/doctl-1.94.0-linux-amd64.tar.gz | tar -xzv"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Requirements check passed${NC}"
}

# Build and test locally first
build_and_test() {
    echo "🔨 Building application locally..."
    
    # Build React frontend
    echo "📦 Building React frontend..."
    cd frontend/react-admin
    npm install
    npm run build
    cd ../..
    
    # Build Docker image
    echo "🐳 Building Docker image..."
    docker build -f Dockerfile.production -t medisense:latest .
    
    # Test the build
    echo "🧪 Testing the build..."
    docker run --rm -d -p 8000:8000 --name medisense-test medisense:latest
    sleep 10
    
    # Check if the app is responding
    if curl -f http://localhost:8000/admin/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Local build test passed${NC}"
        docker stop medisense-test
    else
        echo -e "${RED}❌ Local build test failed${NC}"
        docker stop medisense-test
        exit 1
    fi
}

# Deploy to DigitalOcean
deploy_to_do() {
    echo "☁️ Deploying to DigitalOcean..."
    
    # Check if app already exists
    APP_NAME="medisense-app"
    
    if doctl apps list | grep -q "$APP_NAME"; then
        echo "📝 Updating existing app..."
        doctl apps update $(doctl apps list --format ID --no-header | head -1) --spec .do/app.yaml
    else
        echo "🆕 Creating new app..."
        doctl apps create --spec .do/app.yaml
    fi
    
    echo -e "${GREEN}✅ Deployment initiated!${NC}"
    echo "🔗 Check your app status at: https://cloud.digitalocean.com/apps"
}

# Main deployment flow
main() {
    echo "🏥 MediSense DigitalOcean Deployment"
    echo "===================================="
    
    # Prompt for confirmation
    read -p "Are you ready to deploy to DigitalOcean? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
    
    check_requirements
    build_and_test
    deploy_to_do
    
    echo ""
    echo -e "${GREEN}🎉 Deployment complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Set up your environment variables in DigitalOcean dashboard"
    echo "2. Configure your database connection"
    echo "3. Set up your custom domain (optional)"
    echo "4. Run database migrations: doctl apps logs <app-id>"
}

# Run main function
main "$@"
