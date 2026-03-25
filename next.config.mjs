/** @type {import('next').NextConfig} */
const nextConfig = {
    // Rewrites to handle Python API if needed locally
    async rewrites() {
        return [
            {
                source: '/api/python/:path*',
                destination: 'http://backend:8000/api/python/:path*',
            },
        ]
    },
};

export default nextConfig;
